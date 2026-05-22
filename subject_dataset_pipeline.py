#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps


DEFAULT_DATASET_PATH = Path(
    "/home/yanai-lab/xiong-p/qwen/subject_driven_datasets/dataset_qwen_pe_fixed_subject_driven_no_remove.json"
)
DEFAULT_IMAGE_BASE = Path("/home/yanai-lab/xiong-p/qwen/picobanana/openimages")
DEFAULT_PROCESSING_LOG_PATH = Path(
    "/home/yanai-lab/xiong-p/qwen/picobanana/openimages/pico_sam_output_ALL_20251206_032609/processing_log.json"
)
DEFAULT_OUTPUT_ROOT = Path("/home/yanai-lab/xiong-p/qwen/subject_dataset_runs")
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_VLM_MODEL = "Qwen/Qwen3.6-27B-FP8"
DEFAULT_VLM_URL = "http://127.0.0.1:30000/v1/responses"
DEFAULT_STEPS = 50
DEFAULT_SEED_BASE = 20260523
DEFAULT_TARGET_SIZE = 1024
DEFAULT_VLM_TIMEOUT = 600
DEFAULT_VLM_MAX_OUTPUT_TOKENS = 256
DEFAULT_WORKER_COUNT = 6
LOG_INFO = "INFO"
LOG_WARN = "WARN"
LOG_ERROR = "ERROR"
LOG_DEBUG = "DEBUG"

NEGATIVE_PROMPT = (
    "blurry, out of focus, low resolution, low detail, smeared texture, motion blur, "
    "deformed object, distorted shape, broken geometry, duplicate object, bad composition, "
    "cropped subject, washed out image, same viewpoint as the input image, "
    "same camera angle as the input image"
)

PROMPT_BANK = [
    {"background_prompt_id": "tabletop_001", "scene_background": "a clean wooden table in a sunlit room", "target_scene": "the sunlit tabletop scene", "category": "tabletop"},
    {"background_prompt_id": "tabletop_002", "scene_background": "a minimalist white desk near a large window with soft daylight", "target_scene": "the minimalist desk scene", "category": "tabletop"},
    {"background_prompt_id": "indoor_001", "scene_background": "a modern living room with soft morning light", "target_scene": "the living room scene", "category": "indoor"},
    {"background_prompt_id": "indoor_002", "scene_background": "a clean kitchen interior with natural daylight", "target_scene": "the kitchen interior scene", "category": "indoor"},
    {"background_prompt_id": "nature_001", "scene_background": "a sandy beach at sunset", "target_scene": "the beach scene at sunset", "category": "outdoor_nature"},
    {"background_prompt_id": "nature_002", "scene_background": "a grassy outdoor field under soft daylight", "target_scene": "the grassy outdoor scene", "category": "outdoor_nature"},
    {"background_prompt_id": "street_001", "scene_background": "a rainy neon-lit alley at night", "target_scene": "the rainy neon-lit alley", "category": "street"},
    {"background_prompt_id": "street_002", "scene_background": "a quiet city sidewalk in soft afternoon light", "target_scene": "the city sidewalk scene", "category": "street"},
    {"background_prompt_id": "display_001", "scene_background": "a clean exhibition pedestal under studio lighting", "target_scene": "the exhibition display scene", "category": "display"},
    {"background_prompt_id": "display_002", "scene_background": "a product display platform with soft controlled shadows", "target_scene": "the product display scene", "category": "display"},
    {"background_prompt_id": "studio_001", "scene_background": "a seamless white studio background with soft lighting", "target_scene": "the white studio setting", "category": "studio"},
    {"background_prompt_id": "studio_002", "scene_background": "a neutral gray studio backdrop with soft shadows", "target_scene": "the gray studio setting", "category": "studio"},
]

VIEWPOINT_BANK = [
    {
        "viewpoint_id": "front_center",
        "camera_instruction": "Show the subject from a clean front-facing view with the subject centered in frame, clearly re-rendered from the front.",
    },
    {
        "viewpoint_id": "front_left_three_quarter",
        "camera_instruction": "Show the subject from a strong front-left three-quarter view so a clear amount of its left side is visible.",
    },
    {
        "viewpoint_id": "front_right_three_quarter",
        "camera_instruction": "Show the subject from a strong front-right three-quarter view so a clear amount of its right side is visible.",
    },
    {
        "viewpoint_id": "left_side",
        "camera_instruction": "Show the subject from a clear left-side profile view with the side silhouette strongly visible.",
    },
    {
        "viewpoint_id": "right_side",
        "camera_instruction": "Show the subject from a clear right-side profile view with the side silhouette strongly visible.",
    },
    {
        "viewpoint_id": "slight_top_down",
        "camera_instruction": "Show the subject from an obvious top-down view while keeping it large in frame.",
    },
    {
        "viewpoint_id": "slight_low_angle",
        "camera_instruction": "Show the subject from an obvious low-angle view while keeping it large in frame.",
    },
    {
        "viewpoint_id": "back_view",
        "camera_instruction": "Show the subject clearly from the back so the rear side is the dominant visible view, with the back silhouette strongly visible.",
    },
]

FIXED_PROMPT_PAIRS = [
    ("tabletop_001", "front_right_three_quarter"),
    ("display_001", "slight_top_down"),
    ("nature_002", "left_side"),
    ("street_001", "front_center"),
    ("studio_002", "right_side"),
    ("indoor_002", "slight_low_angle"),
    ("studio_002", "back_view"),
]

SUBJECT_CLEANING_INSTRUCTION_TEMPLATE = (
    "You are cleaning a subject description for subject-driven image generation. "
    "Use both the input image and the long raw subject text. "
    "Extract only the core subject identity and key visual attributes that belong to the object itself. "
    "Remove all background, location, scene, relative position, human interactions, environment, and context words. "
    "Return exactly one short prompt in plain text as a single sentence. "
    "The output must describe only the object and its core components or appearance. "
    "Do not explain. Do not add prefixes. Do not output reasoning.\n\n"
    "Long raw subject text: {raw_subject_text}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-driven dataset pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    coordinator = subparsers.add_parser("coordinator", help="Build manifests and launch workers.")
    coordinator.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    coordinator.add_argument("--image-base", default=str(DEFAULT_IMAGE_BASE))
    coordinator.add_argument("--processing-log-path", default=str(DEFAULT_PROCESSING_LOG_PATH))
    coordinator.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    coordinator.add_argument("--run-name", required=True)
    coordinator.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    coordinator.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    coordinator.add_argument("--subject-offset", type=int, default=0)
    coordinator.add_argument("--subject-limit", type=int, default=None)
    coordinator.add_argument("--prompt-limit", type=int, default=None)
    coordinator.add_argument("--gen-gpus", default="2,3,4,5,6,7")
    coordinator.add_argument("--vlm-url", default=DEFAULT_VLM_URL)
    coordinator.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    coordinator.add_argument("--vlm-timeout", type=int, default=DEFAULT_VLM_TIMEOUT)
    coordinator.add_argument("--vlm-max-output-tokens", type=int, default=DEFAULT_VLM_MAX_OUTPUT_TOKENS)
    coordinator.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    coordinator.add_argument("--resume", action="store_true")
    coordinator.add_argument("--dry-run", action="store_true")

    worker = subparsers.add_parser("worker", help="Run diffusers generation on a single GPU.")
    worker.add_argument("--manifest-path", required=True)
    worker.add_argument("--worker-rank", type=int, required=True)
    worker.add_argument("--num-workers", type=int, required=True)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_prompt(
    scene_background: str,
    target_scene: str,
    subject_description: str | None = None,
    camera_instruction: str | None = None,
) -> str:
    prompt_parts = []
    if subject_description:
        prompt_parts.append(f"The subject is {subject_description}.")
    prompt_parts.extend(
        [
            "Keep the same subject identity from the input image.",
            "Preserve the same object identity, silhouette, colors, texture, markings, and proportions.",
            "Keep the subject as the main focus and make it occupy most of the frame.",
            f"Place the subject in a photorealistic scene with {scene_background}.",
            f"Only change the environment, lighting, and composition so the subject appears naturally placed in {target_scene}.",
            "Do not keep the original camera angle from the input image.",
            "Render the subject from a clearly different viewpoint that matches the requested camera angle.",
        ]
    )
    prompt = " ".join(prompt_parts)
    if camera_instruction:
        prompt += f" Change the camera viewpoint to match this requested angle: {camera_instruction}"
    return prompt


def preprocess_image(image: Image.Image, target_size: int = DEFAULT_TARGET_SIZE) -> tuple[Image.Image, int, int]:
    contained = ImageOps.contain(image, (target_size, target_size), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - contained.width) // 2
    offset_y = (target_size - contained.height) // 2
    canvas.paste(contained, (offset_x, offset_y))
    return canvas, target_size, target_size


def resolve_prompt_specs(prompt_limit: int | None = None) -> list[dict[str, Any]]:
    bg_by_id = {item["background_prompt_id"]: item for item in PROMPT_BANK}
    viewpoint_by_id = {item["viewpoint_id"]: item for item in VIEWPOINT_BANK}

    pairs = FIXED_PROMPT_PAIRS if prompt_limit is None else FIXED_PROMPT_PAIRS[:prompt_limit]
    specs = []
    for prompt_index, (background_prompt_id, viewpoint_id) in enumerate(pairs):
        background = bg_by_id[background_prompt_id]
        viewpoint = viewpoint_by_id[viewpoint_id]
        specs.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": f"{prompt_index:02d}_{background_prompt_id}_{viewpoint_id}",
                "background_prompt_id": background_prompt_id,
                "viewpoint_id": viewpoint_id,
                "scene_background": background["scene_background"],
                "target_scene": background["target_scene"],
                "category": background["category"],
                "camera_instruction": viewpoint["camera_instruction"],
            }
        )
    return specs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def log_event(
    event: str,
    *,
    log_path: Path | None = None,
    level: str = LOG_INFO,
    message: str | None = None,
    **fields: Any,
) -> None:
    timestamp = now_ts()
    payload = {"timestamp": timestamp, "level": level, "event": event, **fields}
    if message:
        payload["message"] = message
    if log_path is not None:
        append_jsonl(log_path, payload)

    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    text = message or event
    if details:
        text = f"{text} | {details}"
    stream = sys.stderr if level in {LOG_WARN, LOG_ERROR} else sys.stdout
    print(f"[{timestamp}] [{level}] {event}: {text}", file=stream, flush=True)


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_item_idx(sample: dict[str, Any]) -> int | None:
    if isinstance(sample.get("item_idx"), int):
        return sample["item_idx"]
    candidates = [sample.get("ref_gt_crop"), sample.get("image"), sample.get("ref_gt")]
    for value in candidates:
        if not value:
            continue
        stem = Path(value).stem
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    return None


def load_processing_log_index(processing_log_path: Path) -> dict[int, dict[str, Any]]:
    with processing_log_path.open() as f:
        rows = json.load(f)
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        item_idx = row.get("item_idx")
        if isinstance(item_idx, int):
            indexed[item_idx] = row
    return indexed


def select_subject_concepts_for_long_text(change_concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    additions = [concept for concept in change_concepts if not concept.get("is_remove", False)]
    removals = [concept for concept in change_concepts if concept.get("is_remove", False)]
    if additions:
        return additions, "target_subject"
    if removals:
        return removals, "source_subject"
    return [], None


def build_raw_long_subject_description(change_concepts: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    chosen_concepts, subject_role = select_subject_concepts_for_long_text(change_concepts)
    if not chosen_concepts:
        return None, subject_role

    descriptions = []
    seen = set()
    for concept in chosen_concepts:
        total_prompt = (concept.get("total_prompt") or "").strip()
        if total_prompt and total_prompt not in seen:
            descriptions.append(total_prompt)
            seen.add(total_prompt)

    if descriptions:
        return " and ".join(descriptions), subject_role

    fallback = []
    for concept in chosen_concepts:
        for sub_prompt in concept.get("sub_prompts_generated", []):
            normalized = sub_prompt.strip()
            if normalized and normalized not in seen:
                fallback.append(normalized)
                seen.add(normalized)

    if fallback:
        return " and ".join(fallback), subject_role
    return None, subject_role


def normalize_prompt_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    lowered = cleaned.lower()
    for prefix in ("prompt:", "subject:", "output:"):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned


def encode_image_as_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_subject_cleaning_instruction(raw_subject_text: str) -> str:
    return SUBJECT_CLEANING_INSTRUCTION_TEMPLATE.format(raw_subject_text=raw_subject_text)


def extract_text_from_vlm_response(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    output = payload.get("output", [])
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text", "").strip():
                return content["text"].strip()
            if content.get("type") == "text" and content.get("text", "").strip():
                return content["text"].strip()
    raise ValueError("No text content found in VLM response payload.")


def clean_subject_with_vlm(
    image_path: Path,
    raw_subject_text: str,
    vlm_url: str,
    vlm_model: str,
    timeout: int,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    image_url = encode_image_as_data_url(image_path)
    instruction = build_subject_cleaning_instruction(raw_subject_text)
    payload = {
        "model": vlm_model,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ],
    }
    response = requests.post(vlm_url, json=payload, timeout=timeout)
    response.raise_for_status()
    response_payload = response.json()
    text = normalize_prompt_text(extract_text_from_vlm_response(response_payload))
    return text, response_payload


def make_run_paths(output_root: Path, run_name: str) -> dict[str, Path]:
    run_dir = output_root / run_name
    metadata_dir = run_dir / "metadata"
    images_dir = run_dir / "images"
    logs_dir = run_dir / "logs"
    return {
        "run_dir": run_dir,
        "metadata_dir": metadata_dir,
        "images_dir": images_dir,
        "logs_dir": logs_dir,
        "vlm_cache_dir": metadata_dir / "vlm_subject_cache",
        "prompt_bank": metadata_dir / "prompt_bank.json",
        "subject_records": metadata_dir / "subject_records.json",
        "generation_log": metadata_dir / "subject_driven_generation_log.jsonl",
        "manifest": metadata_dir / "subject_driven_manifest.json",
        "dataset_all": run_dir / "dataset_qwen_subject_driven_all.json",
    }


def load_existing_subject_records(subject_records_path: Path) -> dict[tuple[int, int | None], dict[str, Any]]:
    if not subject_records_path.exists():
        return {}
    with subject_records_path.open() as f:
        records = json.load(f)
    indexed: dict[tuple[int, int | None], dict[str, Any]] = {}
    for record in records:
        indexed[(record["subject_idx"], record.get("item_idx"))] = record
    return indexed


def deterministic_seeds(seed_base: int, item_idx: int | None, prompt_index: int) -> list[int]:
    stable_item_idx = item_idx if item_idx is not None else 0
    rng = random.Random(seed_base + stable_item_idx + prompt_index)
    return [0, *rng.sample(range(1, 2**31 - 1), 2)]


def output_paths_for_task(
    run_dir: Path,
    item_idx: int | None,
    subject_idx: int,
    prompt_spec: dict[str, Any],
    seed: int,
) -> tuple[Path, Path, str, str]:
    stable_item_idx = item_idx if item_idx is not None else subject_idx
    item_dir = run_dir / "images" / f"item_{stable_item_idx:06d}"
    stem = (
        f"sd_{stable_item_idx:06d}_s{subject_idx:06d}_"
        f"{prompt_spec['background_prompt_id']}_{prompt_spec['viewpoint_id']}_seed{seed}"
    )
    image_path = item_dir / f"{stem}.png"
    metadata_path = item_dir / f"{stem}.json"
    image_rel_path = image_path.relative_to(run_dir).as_posix()
    metadata_rel_path = metadata_path.relative_to(run_dir).as_posix()
    return image_path, metadata_path, image_rel_path, metadata_rel_path


def build_prompt_records(
    run_dir: Path,
    subject_idx: int,
    item_idx: int | None,
    cleaned_subject_description: str | None,
    prompt_specs: list[dict[str, Any]],
    seed_base: int,
) -> list[dict[str, Any]]:
    prompt_records = []
    for prompt_spec in prompt_specs:
        prompt = None
        if cleaned_subject_description:
            prompt = build_prompt(
                prompt_spec["scene_background"],
                prompt_spec["target_scene"],
                cleaned_subject_description,
                prompt_spec["camera_instruction"],
            )
        seeds = deterministic_seeds(seed_base, item_idx, prompt_spec["prompt_index"])
        outputs = []
        for seed in seeds:
            image_path, metadata_path, image_rel_path, metadata_rel_path = output_paths_for_task(
                run_dir, item_idx, subject_idx, prompt_spec, seed
            )
            outputs.append(
                {
                    "seed": seed,
                    "image_rel_path": image_rel_path,
                    "metadata_rel_path": metadata_rel_path,
                }
            )
        prompt_records.append(
            {
                **prompt_spec,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "seeds": seeds,
                "outputs": outputs,
            }
        )
    return prompt_records


def build_subject_record(
    sample: dict[str, Any],
    subject_idx: int,
    item_idx: int | None,
    raw_subject_description: str | None,
    cleaned_subject_description: str | None,
    subject_role: str | None,
    prompt_specs: list[dict[str, Any]],
    run_dir: Path,
    seed_base: int,
    steps: int,
    status: str,
    instruction: str | None,
    error_message: str | None = None,
    vlm_response_text: str | None = None,
) -> dict[str, Any]:
    prompt_records: list[dict[str, Any]] = []
    prompt_records = build_prompt_records(
        run_dir=run_dir,
        subject_idx=subject_idx,
        item_idx=item_idx,
        cleaned_subject_description=cleaned_subject_description,
        prompt_specs=prompt_specs,
        seed_base=seed_base,
    )
    return {
        "subject_idx": subject_idx,
        "item_idx": item_idx,
        "image": sample["image"],
        "edit_image": sample.get("edit_image", []),
        "ref_gt": sample.get("ref_gt"),
        "ref_gt_crop": sample.get("ref_gt_crop"),
        "back_mask": sample.get("back_mask"),
        "edit_type": sample.get("edit_type"),
        "input_image": sample.get("ref_gt_crop"),
        "raw_subject_description": raw_subject_description,
        "cleaned_subject_description": cleaned_subject_description,
        "subject_role": subject_role,
        "instruction": instruction,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": steps,
        "status": status,
        "error_message": error_message,
        "vlm_response_text": vlm_response_text,
        "prompts": prompt_records,
    }


def update_completion_stats(subject_records: list[dict[str, Any]], run_dir: Path) -> None:
    for subject_record in subject_records:
        planned_count = 0
        completed_count = 0
        missing_outputs: list[str] = []
        if subject_record["status"] not in {"ready", "partial", "complete"}:
            subject_record["planned_count"] = 0
            subject_record["completed_count"] = 0
            subject_record["missing_outputs"] = []
            continue

        for prompt_record in subject_record.get("prompts", []):
            prompt_completed = 0
            prompt_missing: list[str] = []
            outputs = prompt_record.get("outputs", [])
            planned_count += len(outputs)
            for output in outputs:
                image_path = run_dir / output["image_rel_path"]
                metadata_path = run_dir / output["metadata_rel_path"]
                if image_path.exists() and metadata_path.exists():
                    prompt_completed += 1
                    completed_count += 1
                else:
                    prompt_missing.append(output["image_rel_path"])
                    missing_outputs.append(output["image_rel_path"])
            prompt_record["completed_count"] = prompt_completed
            prompt_record["planned_count"] = len(outputs)
            prompt_record["missing_outputs"] = prompt_missing

        subject_record["planned_count"] = planned_count
        subject_record["completed_count"] = completed_count
        subject_record["missing_outputs"] = missing_outputs
        if planned_count == 0:
            subject_record["status"] = "ready"
        elif completed_count == planned_count:
            subject_record["status"] = "complete"
        elif completed_count > 0:
            subject_record["status"] = "partial"
        else:
            subject_record["status"] = "ready"


def build_manifest_payload(
    args: argparse.Namespace,
    paths: dict[str, Path],
    prompt_specs: list[dict[str, Any]],
    subject_records: list[dict[str, Any]],
    worker_count: int,
) -> dict[str, Any]:
    ready_subject_count = sum(record["status"] in {"ready", "partial", "complete"} for record in subject_records)
    failed_subject_count = sum(record["status"] not in {"ready", "partial", "complete"} for record in subject_records)
    planned_image_count = sum(record.get("planned_count", 0) for record in subject_records)
    completed_image_count = sum(record.get("completed_count", 0) for record in subject_records)
    return {
        "run_name": args.run_name,
        "run_dir": str(paths["run_dir"]),
        "dataset_path": str(Path(args.dataset_path)),
        "image_base": str(Path(args.image_base)),
        "processing_log_path": str(Path(args.processing_log_path)),
        "model_id": args.model_id,
        "vlm_model": args.vlm_model,
        "vlm_url": args.vlm_url,
        "vlm_timeout": args.vlm_timeout,
        "vlm_max_output_tokens": args.vlm_max_output_tokens,
        "worker_count": worker_count,
        "gen_gpus": parse_gpu_list(args.gen_gpus),
        "steps": args.steps,
        "seed_base": args.seed_base,
        "negative_prompt": NEGATIVE_PROMPT,
        "target_size": DEFAULT_TARGET_SIZE,
        "subject_offset": args.subject_offset,
        "subject_limit": args.subject_limit,
        "prompt_limit": args.prompt_limit,
        "prompt_bank_path": str(paths["prompt_bank"]),
        "subject_records_path": str(paths["subject_records"]),
        "generation_log_path": str(paths["generation_log"]),
        "dataset_all_path": str(paths["dataset_all"]),
        "vlm_cache_dir": str(paths["vlm_cache_dir"]),
        "created_at": now_ts(),
        "selected_subject_count": len(subject_records),
        "ready_subject_count": ready_subject_count,
        "failed_subject_count": failed_subject_count,
        "planned_image_count": planned_image_count,
        "completed_image_count": completed_image_count,
        "prompt_bank": prompt_specs,
        "subjects": subject_records,
    }


def build_flat_dataset(run_dir: Path, subject_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset_rows = []
    for subject_record in subject_records:
        for prompt_record in subject_record.get("prompts", []):
            for output in prompt_record.get("outputs", []):
                metadata_path = run_dir / output["metadata_rel_path"]
                if not metadata_path.exists():
                    continue
                with metadata_path.open() as f:
                    meta = json.load(f)
                dataset_rows.append(
                    {
                        "image": meta["image_rel_path"],
                        "edit_image": meta["edit_image"],
                        "ref_gt": meta["ref_gt"],
                        "ref_gt_crop": meta["ref_gt_crop"],
                        "back_mask": meta.get("back_mask"),
                        "edit_type": meta.get("edit_type"),
                        "prompt": meta["prompt"],
                        "negative_prompt": meta["negative_prompt"],
                        "cleaned_subject_description": meta["cleaned_subject_description"],
                        "raw_subject_description": meta["raw_subject_description"],
                        "background_prompt_id": meta["background_prompt_id"],
                        "viewpoint_id": meta["viewpoint_id"],
                        "scene_background": meta["scene_background"],
                        "target_scene": meta["target_scene"],
                        "camera_instruction": meta["camera_instruction"],
                        "seed": meta["seed"],
                        "steps": meta["steps"],
                        "item_idx": meta["item_idx"],
                        "subject_idx": meta["subject_idx"],
                        "prompt_index": meta["prompt_index"],
                        "prompt_id": meta["prompt_id"],
                    }
                )
    dataset_rows.sort(key=lambda row: (row["subject_idx"], row["prompt_index"], row["seed"]))
    return dataset_rows


def parse_gpu_list(raw: str) -> list[str]:
    gpu_ids = [token.strip() for token in raw.split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("Expected at least one GPU id.")
    return gpu_ids


def launch_workers(args: argparse.Namespace, manifest_path: Path, gpu_ids: list[str]) -> list[int]:
    processes = []
    command_base = [sys.executable, str(Path(__file__).resolve()), "worker", "--manifest-path", str(manifest_path), "--num-workers", str(len(gpu_ids))]
    for worker_rank, gpu_id in enumerate(gpu_ids):
        cmd = [*command_base, "--worker-rank", str(worker_rank)]
        if args.resume:
            cmd.append("--resume")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("_CHECK_PEFT", "0")
        print(
            f"[{now_ts()}] [INFO] worker_launch: rank={worker_rank} gpu={gpu_id} command={' '.join(cmd)}",
            flush=True,
        )
        processes.append(subprocess.Popen(cmd, env=env))

    exit_codes = []
    for worker_rank, process in enumerate(processes):
        exit_code = process.wait()
        exit_codes.append(exit_code)
        level = LOG_INFO if exit_code == 0 else LOG_ERROR
        stream = sys.stdout if exit_code == 0 else sys.stderr
        print(f"[{now_ts()}] [{level}] worker_exit: rank={worker_rank} exit_code={exit_code}", file=stream, flush=True)
    return exit_codes


def maybe_reuse_cleaned_subject(existing_record: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not existing_record:
        return None, None
    cleaned = existing_record.get("cleaned_subject_description")
    response_text = existing_record.get("vlm_response_text")
    if cleaned:
        return cleaned, response_text
    return None, None


def subject_cache_path(vlm_cache_dir: Path, subject_record: dict[str, Any]) -> Path:
    item_idx = subject_record.get("item_idx")
    item_part = f"i{item_idx:06d}" if isinstance(item_idx, int) else "inoitem"
    return vlm_cache_dir / f"s{subject_record['subject_idx']:06d}_{item_part}.json"


def read_cached_subject_cleaning(cache_path: Path) -> tuple[str | None, str | None]:
    if not cache_path.exists():
        return None, None
    with cache_path.open() as f:
        payload = json.load(f)
    cleaned = payload.get("cleaned_subject_description")
    response_text = payload.get("vlm_response_text")
    if isinstance(cleaned, str) and cleaned.strip():
        return cleaned.strip(), response_text if isinstance(response_text, str) else None
    return None, None


def write_cached_subject_cleaning(
    cache_path: Path,
    subject_record: dict[str, Any],
    cleaned_subject_description: str,
    vlm_response_text: str | None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    payload = {
        "subject_idx": subject_record["subject_idx"],
        "item_idx": subject_record.get("item_idx"),
        "raw_subject_description": subject_record.get("raw_subject_description"),
        "cleaned_subject_description": cleaned_subject_description,
        "vlm_response_text": vlm_response_text,
        "cleaned_at": now_ts(),
    }
    write_json(tmp_path, payload)
    tmp_path.replace(cache_path)


def apply_cleaned_subject_to_record(
    subject_record: dict[str, Any],
    cleaned_subject_description: str,
    vlm_response_text: str | None,
) -> None:
    subject_record["cleaned_subject_description"] = cleaned_subject_description
    subject_record["vlm_response_text"] = vlm_response_text
    for prompt_record in subject_record.get("prompts", []):
        prompt_record["prompt"] = build_prompt(
            prompt_record["scene_background"],
            prompt_record["target_scene"],
            cleaned_subject_description,
            prompt_record["camera_instruction"],
        )


def acquire_cache_lock(lock_path: Path, timeout_seconds: int = 3600) -> None:
    start_ts = time.time()
    last_wait_log_ts = 0.0
    while True:
        try:
            os.mkdir(lock_path)
            return
        except FileExistsError:
            now = time.time()
            if now - last_wait_log_ts >= 30:
                print(
                    f"[{now_ts()}] [DEBUG] cache_lock_wait: path={lock_path} waited_seconds={now - start_ts:.1f}",
                    flush=True,
                )
                last_wait_log_ts = now
            if time.time() - start_ts > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for VLM cache lock: {lock_path}")
            try:
                if time.time() - lock_path.stat().st_mtime > timeout_seconds:
                    os.rmdir(lock_path)
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                pass
            time.sleep(2)


def release_cache_lock(lock_path: Path) -> None:
    try:
        os.rmdir(lock_path)
    except FileNotFoundError:
        pass


def ensure_subject_cleaned_for_worker(
    subject_record: dict[str, Any],
    manifest: dict[str, Any],
    image_base: Path,
    generation_log_path: Path,
    worker_rank: int,
) -> tuple[str, str | None]:
    existing_cleaned = subject_record.get("cleaned_subject_description")
    if isinstance(existing_cleaned, str) and existing_cleaned.strip():
        log_event(
            "lazy_clean_manifest_hit",
            log_path=generation_log_path,
            level=LOG_DEBUG,
            worker_rank=worker_rank,
            subject_idx=subject_record["subject_idx"],
            item_idx=subject_record.get("item_idx"),
        )
        return existing_cleaned.strip(), subject_record.get("vlm_response_text")

    vlm_cache_dir = Path(manifest.get("vlm_cache_dir", Path(manifest["run_dir"]) / "metadata" / "vlm_subject_cache"))
    cache_path = subject_cache_path(vlm_cache_dir, subject_record)
    cleaned, response_text = read_cached_subject_cleaning(cache_path)
    if cleaned:
        apply_cleaned_subject_to_record(subject_record, cleaned, response_text)
        log_event(
            "lazy_clean_cache_hit",
            log_path=generation_log_path,
            level=LOG_DEBUG,
            worker_rank=worker_rank,
            subject_idx=subject_record["subject_idx"],
            item_idx=subject_record.get("item_idx"),
            cache_path=str(cache_path),
        )
        return cleaned, response_text

    raw_subject_text = subject_record.get("raw_subject_description")
    if not isinstance(raw_subject_text, str) or not raw_subject_text.strip():
        raise ValueError("Missing raw subject description for lazy VLM cleaning.")

    input_image_path = image_base / subject_record["ref_gt_crop"]
    lock_path = cache_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_wait_start = time.time()
    log_event(
        "lazy_clean_lock_wait",
        log_path=generation_log_path,
        level=LOG_DEBUG,
        worker_rank=worker_rank,
        subject_idx=subject_record["subject_idx"],
        item_idx=subject_record.get("item_idx"),
        lock_path=str(lock_path),
    )
    acquire_cache_lock(lock_path)
    lock_wait_seconds = time.time() - lock_wait_start
    log_event(
        "lazy_clean_lock_acquired",
        log_path=generation_log_path,
        level=LOG_DEBUG,
        worker_rank=worker_rank,
        subject_idx=subject_record["subject_idx"],
        item_idx=subject_record.get("item_idx"),
        lock_wait_seconds=round(lock_wait_seconds, 3),
    )
    try:
        cleaned, response_text = read_cached_subject_cleaning(cache_path)
        if not cleaned:
            clean_start = time.time()
            log_event(
                "lazy_clean_started",
                log_path=generation_log_path,
                worker_rank=worker_rank,
                subject_idx=subject_record["subject_idx"],
                item_idx=subject_record.get("item_idx"),
                input_image_path=str(input_image_path),
                raw_subject_chars=len(raw_subject_text),
            )
            cleaned, response_payload = clean_subject_with_vlm(
                image_path=input_image_path,
                raw_subject_text=raw_subject_text,
                vlm_url=manifest["vlm_url"],
                vlm_model=manifest["vlm_model"],
                timeout=int(manifest.get("vlm_timeout", DEFAULT_VLM_TIMEOUT)),
                max_output_tokens=int(manifest.get("vlm_max_output_tokens", DEFAULT_VLM_MAX_OUTPUT_TOKENS)),
            )
            response_text = extract_text_from_vlm_response(response_payload)
            if not cleaned:
                raise ValueError("VLM cleaning returned an empty subject description.")
            write_cached_subject_cleaning(cache_path, subject_record, cleaned, response_text)
            log_event(
                "lazy_clean_completed",
                log_path=generation_log_path,
                worker_rank=worker_rank,
                subject_idx=subject_record["subject_idx"],
                item_idx=subject_record.get("item_idx"),
                cleaned_subject_description=cleaned,
                duration_seconds=round(time.time() - clean_start, 3),
                cache_path=str(cache_path),
            )
        else:
            log_event(
                "lazy_clean_cache_hit_after_lock",
                log_path=generation_log_path,
                level=LOG_DEBUG,
                worker_rank=worker_rank,
                subject_idx=subject_record["subject_idx"],
                item_idx=subject_record.get("item_idx"),
                cache_path=str(cache_path),
            )
    finally:
        release_cache_lock(lock_path)
        log_event(
            "lazy_clean_lock_released",
            log_path=generation_log_path,
            level=LOG_DEBUG,
            worker_rank=worker_rank,
            subject_idx=subject_record["subject_idx"],
            item_idx=subject_record.get("item_idx"),
        )

    apply_cleaned_subject_to_record(subject_record, cleaned, response_text)
    return cleaned, response_text


def sync_subject_records_from_vlm_cache(paths: dict[str, Path], subject_records: list[dict[str, Any]]) -> None:
    for subject_record in subject_records:
        if subject_record.get("status") not in {"ready", "partial", "complete"}:
            continue
        cache_path = subject_cache_path(paths["vlm_cache_dir"], subject_record)
        cleaned, response_text = read_cached_subject_cleaning(cache_path)
        if cleaned:
            apply_cleaned_subject_to_record(subject_record, cleaned, response_text)


def run_coordinator(args: argparse.Namespace) -> int:
    coordinator_start = time.time()
    output_root = Path(args.output_root)
    paths = make_run_paths(output_root, args.run_name)
    prompt_specs = resolve_prompt_specs(args.prompt_limit)
    worker_gpu_ids = parse_gpu_list(args.gen_gpus)
    worker_count = len(worker_gpu_ids)

    if worker_count != DEFAULT_WORKER_COUNT:
        log_event(
            "worker_count_unexpected",
            level=LOG_WARN,
            message=f"expected 6 generation GPUs, got {worker_count}: {worker_gpu_ids}",
            expected_worker_count=DEFAULT_WORKER_COUNT,
            actual_worker_count=worker_count,
            gen_gpus=",".join(worker_gpu_ids),
        )

    if paths["run_dir"].exists() and not args.resume:
        disallowed_entries = [entry for entry in paths["run_dir"].iterdir() if entry.name != "logs"]
        if disallowed_entries:
            raise FileExistsError(f"Run directory already exists and is not empty: {paths['run_dir']}")

    for path in (paths["run_dir"], paths["metadata_dir"], paths["images_dir"], paths["logs_dir"], paths["vlm_cache_dir"]):
        path.mkdir(parents=True, exist_ok=True)

    log_event(
        "coordinator_started",
        log_path=paths["generation_log"],
        run_name=args.run_name,
        run_dir=str(paths["run_dir"]),
        dataset_path=str(Path(args.dataset_path)),
        image_base=str(Path(args.image_base)),
        processing_log_path=str(Path(args.processing_log_path)),
        subject_offset=args.subject_offset,
        subject_limit=args.subject_limit,
        prompt_limit=args.prompt_limit,
        gen_gpus=",".join(worker_gpu_ids),
        resume=args.resume,
        dry_run=args.dry_run,
    )

    write_json(paths["prompt_bank"], prompt_specs)

    with Path(args.dataset_path).open() as f:
        all_samples = json.load(f)

    selected_samples = all_samples[args.subject_offset :]
    if args.subject_limit is not None:
        selected_samples = selected_samples[: args.subject_limit]

    existing_records_by_key = load_existing_subject_records(paths["subject_records"]) if args.resume else {}
    processing_log_index = load_processing_log_index(Path(args.processing_log_path))
    log_event(
        "coordinator_inputs_loaded",
        log_path=paths["generation_log"],
        total_samples=len(all_samples),
        selected_samples=len(selected_samples),
        processing_log_rows=len(processing_log_index),
        existing_records=len(existing_records_by_key),
        prompt_specs=len(prompt_specs),
    )

    subject_records = []
    for local_offset, sample in enumerate(selected_samples):
        subject_idx = args.subject_offset + local_offset
        item_idx = parse_item_idx(sample)
        existing_record = existing_records_by_key.get((subject_idx, item_idx))

        processing_row = processing_log_index.get(item_idx) if item_idx is not None else None
        change_concepts = processing_row.get("change_concepts", []) if processing_row else []
        instruction = processing_row.get("instruction") if processing_row else None
        raw_subject_description, subject_role = build_raw_long_subject_description(change_concepts)
        input_image_path = Path(args.image_base) / sample["ref_gt_crop"]

        if not input_image_path.exists():
            error_message = f"Missing input image: {input_image_path}"
            record = build_subject_record(
                sample=sample,
                subject_idx=subject_idx,
                item_idx=item_idx,
                raw_subject_description=raw_subject_description,
                cleaned_subject_description=None,
                subject_role=subject_role,
                prompt_specs=prompt_specs,
                run_dir=paths["run_dir"],
                seed_base=args.seed_base,
                steps=args.steps,
                status="input_image_missing",
                instruction=instruction,
                error_message=error_message,
            )
            subject_records.append(record)
            log_event(
                "subject_skipped",
                log_path=paths["generation_log"],
                level=LOG_WARN,
                subject_idx=subject_idx,
                item_idx=item_idx,
                reason=error_message,
            )
            continue

        if not raw_subject_description:
            error_message = "Could not build raw subject description from processing log."
            record = build_subject_record(
                sample=sample,
                subject_idx=subject_idx,
                item_idx=item_idx,
                raw_subject_description=None,
                cleaned_subject_description=None,
                subject_role=subject_role,
                prompt_specs=prompt_specs,
                run_dir=paths["run_dir"],
                seed_base=args.seed_base,
                steps=args.steps,
                status="raw_subject_missing",
                instruction=instruction,
                error_message=error_message,
            )
            subject_records.append(record)
            log_event(
                "subject_skipped",
                log_path=paths["generation_log"],
                level=LOG_WARN,
                subject_idx=subject_idx,
                item_idx=item_idx,
                reason=error_message,
            )
            continue

        cleaned_subject_description, vlm_response_text = maybe_reuse_cleaned_subject(existing_record)
        if cleaned_subject_description:
            write_cached_subject_cleaning(
                subject_cache_path(paths["vlm_cache_dir"], {"subject_idx": subject_idx, "item_idx": item_idx, "raw_subject_description": raw_subject_description}),
                {"subject_idx": subject_idx, "item_idx": item_idx, "raw_subject_description": raw_subject_description},
                cleaned_subject_description,
                vlm_response_text,
            )
            log_event(
                "subject_reused_clean_cache_seeded",
                log_path=paths["generation_log"],
                level=LOG_DEBUG,
                subject_idx=subject_idx,
                item_idx=item_idx,
            )

        record = build_subject_record(
            sample=sample,
            subject_idx=subject_idx,
            item_idx=item_idx,
            raw_subject_description=raw_subject_description,
            cleaned_subject_description=cleaned_subject_description,
            subject_role=subject_role,
            prompt_specs=prompt_specs,
            run_dir=paths["run_dir"],
            seed_base=args.seed_base,
            steps=args.steps,
            status="ready",
            instruction=instruction,
            vlm_response_text=vlm_response_text,
        )
        subject_records.append(record)
        log_event(
            "subject_manifest_ready",
            log_path=paths["generation_log"],
            level=LOG_DEBUG,
            subject_idx=subject_idx,
            item_idx=item_idx,
            raw_subject_chars=len(raw_subject_description),
            has_reused_cleaned_subject=bool(cleaned_subject_description),
        )

    sync_subject_records_from_vlm_cache(paths, subject_records)
    update_completion_stats(subject_records, paths["run_dir"])
    write_json(paths["subject_records"], subject_records)
    manifest_payload = build_manifest_payload(args, paths, prompt_specs, subject_records, worker_count)
    write_json(paths["manifest"], manifest_payload)
    log_event(
        "manifest_written",
        log_path=paths["generation_log"],
        manifest_path=str(paths["manifest"]),
        selected_subject_count=len(subject_records),
        ready_subject_count=manifest_payload["ready_subject_count"],
        failed_subject_count=manifest_payload["failed_subject_count"],
        planned_image_count=manifest_payload["planned_image_count"],
        completed_image_count=manifest_payload["completed_image_count"],
        lazy_cleaning=True,
    )

    if args.dry_run:
        dataset_rows = build_flat_dataset(paths["run_dir"], subject_records)
        write_json(paths["dataset_all"], dataset_rows)
        log_event(
            "coordinator_dry_run_completed",
            log_path=paths["generation_log"],
            manifest_path=str(paths["manifest"]),
            dataset_all_path=str(paths["dataset_all"]),
            duration_seconds=round(time.time() - coordinator_start, 3),
        )
        return 0

    log_event(
        "workers_launching",
        log_path=paths["generation_log"],
        worker_count=worker_count,
        manifest_path=str(paths["manifest"]),
    )
    exit_codes = launch_workers(args, paths["manifest"], worker_gpu_ids)
    sync_subject_records_from_vlm_cache(paths, subject_records)
    update_completion_stats(subject_records, paths["run_dir"])
    write_json(paths["subject_records"], subject_records)
    manifest_payload = build_manifest_payload(args, paths, prompt_specs, subject_records, worker_count)
    manifest_payload["worker_exit_codes"] = exit_codes
    write_json(paths["manifest"], manifest_payload)
    dataset_rows = build_flat_dataset(paths["run_dir"], subject_records)
    write_json(paths["dataset_all"], dataset_rows)
    log_event(
        "coordinator_completed",
        log_path=paths["generation_log"],
        level=LOG_ERROR if any(code != 0 for code in exit_codes) else LOG_INFO,
        worker_exit_codes=exit_codes,
        completed_image_count=manifest_payload["completed_image_count"],
        planned_image_count=manifest_payload["planned_image_count"],
        dataset_rows=len(dataset_rows),
        duration_seconds=round(time.time() - coordinator_start, 3),
    )

    if any(code != 0 for code in exit_codes):
        log_event(
            "worker_failure_detected",
            log_path=paths["generation_log"],
            level=LOG_WARN,
            message=f"one or more workers exited non-zero: {exit_codes}",
            worker_exit_codes=exit_codes,
        )
        return 1
    return 0


def grouped_worker_tasks(subjects: list[dict[str, Any]], worker_rank: int, num_workers: int) -> list[dict[str, Any]]:
    all_tasks = []
    for subject_record in subjects:
        if subject_record.get("status") not in {"ready", "partial", "complete"}:
            continue
        for prompt_record in subject_record.get("prompts", []):
            for output in prompt_record.get("outputs", []):
                all_tasks.append(
                    {
                        "subject_record": subject_record,
                        "prompt_record": prompt_record,
                        "output": output,
                    }
                )
    return all_tasks[worker_rank::num_workers]


def subject_tasks_need_generation(subject_tasks: list[dict[str, Any]], run_dir: Path) -> bool:
    for task in subject_tasks:
        output = task["output"]
        image_path = run_dir / output["image_rel_path"]
        metadata_path = run_dir / output["metadata_rel_path"]
        if not (image_path.exists() and metadata_path.exists()):
            return True
    return False


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_worker(args: argparse.Namespace) -> int:
    os.environ.setdefault("_CHECK_PEFT", "0")
    from diffusers import DiffusionPipeline
    from diffusers.utils import load_image
    import torch

    manifest_path = Path(args.manifest_path)
    with manifest_path.open() as f:
        manifest = json.load(f)

    run_dir = Path(manifest["run_dir"])
    image_base = Path(manifest["image_base"])
    model_id = manifest["model_id"]
    steps = manifest["steps"]
    generation_log_path = Path(manifest["generation_log_path"])
    subjects = manifest["subjects"]
    tasks = grouped_worker_tasks(subjects, args.worker_rank, args.num_workers)

    log_event(
        "worker_started",
        log_path=generation_log_path,
        worker_rank=args.worker_rank,
        num_workers=args.num_workers,
        run_name=manifest["run_name"],
        run_dir=str(run_dir),
        model_id=model_id,
        assigned_tasks=len(tasks),
        total_subjects=len(subjects),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        dry_run=args.dry_run,
    )

    if args.dry_run:
        log_event(
            "worker_dry_run_completed",
            log_path=generation_log_path,
            worker_rank=args.worker_rank,
            assigned_tasks=len(tasks),
        )
        return 0

    model_load_start = time.time()
    log_event(
        "model_load_started",
        log_path=generation_log_path,
        worker_rank=args.worker_rank,
        model_id=model_id,
    )
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    log_event(
        "model_load_completed",
        log_path=generation_log_path,
        worker_rank=args.worker_rank,
        model_id=model_id,
        duration_seconds=round(time.time() - model_load_start, 3),
    )

    tasks_by_subject: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_subject.setdefault(task["subject_record"]["subject_idx"], []).append(task)

    had_failures = False
    subject_batches = list(tasks_by_subject.items())
    log_event(
        "worker_tasks_grouped",
        log_path=generation_log_path,
        worker_rank=args.worker_rank,
        subject_batches=len(subject_batches),
        assigned_tasks=len(tasks),
    )

    def preclean_subject(subject_tasks: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        subject_record = subject_tasks[0]["subject_record"]
        if not subject_tasks_need_generation(subject_tasks, run_dir):
            log_event(
                "preclean_skipped_existing_subject",
                log_path=generation_log_path,
                level=LOG_DEBUG,
                worker_rank=args.worker_rank,
                subject_idx=subject_record["subject_idx"],
                item_idx=subject_record.get("item_idx"),
            )
            return None, None
        log_event(
            "preclean_submitted",
            log_path=generation_log_path,
            level=LOG_DEBUG,
            worker_rank=args.worker_rank,
            subject_idx=subject_record["subject_idx"],
            item_idx=subject_record.get("item_idx"),
            task_count=len(subject_tasks),
        )
        return ensure_subject_cleaned_for_worker(
            subject_record=subject_record,
            manifest=manifest,
            image_base=image_base,
            generation_log_path=generation_log_path,
            worker_rank=args.worker_rank,
        )

    with ThreadPoolExecutor(max_workers=1) as preclean_executor:
        future_by_index = {}
        if subject_batches:
            future_by_index[0] = preclean_executor.submit(preclean_subject, subject_batches[0][1])
            log_event(
                "preclean_future_scheduled",
                log_path=generation_log_path,
                level=LOG_DEBUG,
                worker_rank=args.worker_rank,
                batch_index=0,
                subject_idx=subject_batches[0][0],
            )

        for batch_index, (subject_idx, subject_tasks) in enumerate(subject_batches):
            subject_start = time.time()
            clean_future = future_by_index.pop(batch_index, None)
            if clean_future is not None:
                try:
                    clean_future.result()
                except Exception as exc:  # noqa: BLE001
                    had_failures = True
                    subject_record = subject_tasks[0]["subject_record"]
                    log_event(
                        "lazy_clean_failed",
                        log_path=generation_log_path,
                        level=LOG_ERROR,
                        worker_rank=args.worker_rank,
                        subject_idx=subject_idx,
                        item_idx=subject_record.get("item_idx"),
                        error=str(exc),
                    )
                    if batch_index + 1 < len(subject_batches):
                        future_by_index[batch_index + 1] = preclean_executor.submit(preclean_subject, subject_batches[batch_index + 1][1])
                        log_event(
                            "preclean_future_scheduled",
                            log_path=generation_log_path,
                            level=LOG_DEBUG,
                            worker_rank=args.worker_rank,
                            batch_index=batch_index + 1,
                            subject_idx=subject_batches[batch_index + 1][0],
                        )
                    continue

            if batch_index + 1 < len(subject_batches):
                future_by_index[batch_index + 1] = preclean_executor.submit(preclean_subject, subject_batches[batch_index + 1][1])
                log_event(
                    "preclean_future_scheduled",
                    log_path=generation_log_path,
                    level=LOG_DEBUG,
                    worker_rank=args.worker_rank,
                    batch_index=batch_index + 1,
                    subject_idx=subject_batches[batch_index + 1][0],
                )

            subject_record = subject_tasks[0]["subject_record"]
            log_event(
                "subject_generation_started",
                log_path=generation_log_path,
                worker_rank=args.worker_rank,
                subject_idx=subject_idx,
                item_idx=subject_record.get("item_idx"),
                task_count=len(subject_tasks),
            )
            input_image_path = image_base / subject_record["ref_gt_crop"]
            if not input_image_path.exists():
                had_failures = True
                log_event(
                    "generation_failed",
                    log_path=generation_log_path,
                    level=LOG_ERROR,
                    worker_rank=args.worker_rank,
                    subject_idx=subject_idx,
                    item_idx=subject_record.get("item_idx"),
                    error=f"Missing input image: {input_image_path}",
                )
                continue

            image_load_start = time.time()
            input_image = load_image(str(input_image_path)).convert("RGB")
            input_image, _, _ = preprocess_image(input_image, DEFAULT_TARGET_SIZE)
            log_event(
                "input_image_loaded",
                log_path=generation_log_path,
                level=LOG_DEBUG,
                worker_rank=args.worker_rank,
                subject_idx=subject_idx,
                item_idx=subject_record.get("item_idx"),
                input_image_path=str(input_image_path),
                processed_width=input_image.width,
                processed_height=input_image.height,
                duration_seconds=round(time.time() - image_load_start, 3),
            )

            generated_count = 0
            skipped_count = 0

            for task in subject_tasks:
                prompt_record = task["prompt_record"]
                output = task["output"]
                seed = output["seed"]
                image_path = run_dir / output["image_rel_path"]
                metadata_path = run_dir / output["metadata_rel_path"]
                if image_path.exists() and metadata_path.exists():
                    skipped_count += 1
                    log_event(
                        "skipped_existing",
                        log_path=generation_log_path,
                        level=LOG_DEBUG,
                        worker_rank=args.worker_rank,
                        subject_idx=subject_idx,
                        item_idx=subject_record.get("item_idx"),
                        prompt_id=prompt_record["prompt_id"],
                        seed=seed,
                        image_rel_path=output["image_rel_path"],
                    )
                    continue

                if not prompt_record.get("prompt"):
                    cleaned_subject_description = subject_record.get("cleaned_subject_description")
                    if not cleaned_subject_description:
                        cleaned_subject_description, _ = ensure_subject_cleaned_for_worker(
                            subject_record=subject_record,
                            manifest=manifest,
                            image_base=image_base,
                            generation_log_path=generation_log_path,
                            worker_rank=args.worker_rank,
                        )
                    prompt_record["prompt"] = build_prompt(
                        prompt_record["scene_background"],
                        prompt_record["target_scene"],
                        cleaned_subject_description,
                        prompt_record["camera_instruction"],
                    )
                    log_event(
                        "prompt_built",
                        log_path=generation_log_path,
                        level=LOG_DEBUG,
                        worker_rank=args.worker_rank,
                        subject_idx=subject_idx,
                        item_idx=subject_record.get("item_idx"),
                        prompt_id=prompt_record["prompt_id"],
                        prompt_chars=len(prompt_record["prompt"]),
                    )

                ensure_parent(image_path)
                try:
                    generation_start = time.time()
                    log_event(
                        "image_generation_started",
                        log_path=generation_log_path,
                        worker_rank=args.worker_rank,
                        subject_idx=subject_idx,
                        item_idx=subject_record.get("item_idx"),
                        prompt_id=prompt_record["prompt_id"],
                        seed=seed,
                        image_rel_path=output["image_rel_path"],
                    )
                    result = pipe(
                        image=input_image,
                        prompt=prompt_record["prompt"],
                        negative_prompt=prompt_record["negative_prompt"],
                        width=input_image.width,
                        height=input_image.height,
                        num_inference_steps=steps,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                    )
                    image = result.images[0]
                    image.save(image_path)
                    generation_seconds = time.time() - generation_start

                    metadata = {
                        "run_name": manifest["run_name"],
                        "subject_idx": subject_record["subject_idx"],
                        "item_idx": subject_record.get("item_idx"),
                        "prompt_index": prompt_record["prompt_index"],
                        "prompt_id": prompt_record["prompt_id"],
                        "image_rel_path": output["image_rel_path"],
                        "image_abs_path": str(image_path),
                        "metadata_rel_path": output["metadata_rel_path"],
                        "metadata_abs_path": str(metadata_path),
                        "image": output["image_rel_path"],
                        "edit_image": [subject_record["ref_gt_crop"]],
                        "ref_gt": subject_record["ref_gt"],
                        "ref_gt_crop": subject_record["ref_gt_crop"],
                        "back_mask": subject_record.get("back_mask"),
                        "edit_type": subject_record.get("edit_type"),
                        "input_image_abs_path": str(input_image_path),
                        "source_image": subject_record["image"],
                        "raw_subject_description": subject_record["raw_subject_description"],
                        "cleaned_subject_description": subject_record["cleaned_subject_description"],
                        "subject_role": subject_record["subject_role"],
                        "instruction": subject_record["instruction"],
                        "background_prompt_id": prompt_record["background_prompt_id"],
                        "viewpoint_id": prompt_record["viewpoint_id"],
                        "scene_background": prompt_record["scene_background"],
                        "target_scene": prompt_record["target_scene"],
                        "camera_instruction": prompt_record["camera_instruction"],
                        "prompt": prompt_record["prompt"],
                        "negative_prompt": prompt_record["negative_prompt"],
                        "seed": seed,
                        "steps": steps,
                        "model_id": model_id,
                        "processed_width": input_image.width,
                        "processed_height": input_image.height,
                        "worker_rank": args.worker_rank,
                        "num_workers": args.num_workers,
                        "generated_at": now_ts(),
                    }
                    write_json(metadata_path, metadata)
                    generated_count += 1
                    log_event(
                        "generated",
                        log_path=generation_log_path,
                        worker_rank=args.worker_rank,
                        subject_idx=subject_idx,
                        item_idx=subject_record.get("item_idx"),
                        prompt_id=prompt_record["prompt_id"],
                        seed=seed,
                        image_rel_path=output["image_rel_path"],
                        metadata_rel_path=output["metadata_rel_path"],
                        duration_seconds=round(generation_seconds, 3),
                    )
                except Exception as exc:  # noqa: BLE001
                    had_failures = True
                    log_event(
                        "generation_failed",
                        log_path=generation_log_path,
                        level=LOG_ERROR,
                        worker_rank=args.worker_rank,
                        subject_idx=subject_idx,
                        item_idx=subject_record.get("item_idx"),
                        prompt_id=prompt_record["prompt_id"],
                        seed=seed,
                        error=str(exc),
                    )
            log_event(
                "subject_generation_completed",
                log_path=generation_log_path,
                worker_rank=args.worker_rank,
                subject_idx=subject_idx,
                item_idx=subject_record.get("item_idx"),
                generated_count=generated_count,
                skipped_count=skipped_count,
                duration_seconds=round(time.time() - subject_start, 3),
            )
    log_event(
        "worker_completed",
        log_path=generation_log_path,
        level=LOG_ERROR if had_failures else LOG_INFO,
        worker_rank=args.worker_rank,
        had_failures=had_failures,
        assigned_tasks=len(tasks),
        subject_batches=len(subject_batches),
    )
    return 1 if had_failures else 0


def main() -> int:
    args = parse_args()
    if args.command == "coordinator":
        return run_coordinator(args)
    if args.command == "worker":
        return run_worker(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
