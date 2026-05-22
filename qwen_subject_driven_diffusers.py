import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image, ImageOps


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
        "camera_instruction": "Show the subject from a clean front-facing view with the subject centered in frame, clearly re-rendered from the front."
    },
    {
        "viewpoint_id": "front_left_three_quarter",
        "camera_instruction": "Show the subject from a strong front-left three-quarter view so a clear amount of its left side is visible."
    },
    {
        "viewpoint_id": "front_right_three_quarter",
        "camera_instruction": "Show the subject from a strong front-right three-quarter view so a clear amount of its right side is visible."
    },
    {
        "viewpoint_id": "left_side",
        "camera_instruction": "Show the subject from a clear left-side profile view with the side silhouette strongly visible."
    },
    {
        "viewpoint_id": "right_side",
        "camera_instruction": "Show the subject from a clear right-side profile view with the side silhouette strongly visible."
    },
    {
        "viewpoint_id": "slight_top_down",
        "camera_instruction": "Show the subject from an obvious top-down view while keeping it large in frame."
    },
    {
        "viewpoint_id": "slight_low_angle",
        "camera_instruction": "Show the subject from an obvious low-angle view while keeping it large in frame."
    },
    {
        "viewpoint_id": "back_view",
        "camera_instruction": "Show the subject clearly from the back so the rear side is the dominant visible view, with the back silhouette strongly visible."
    },
]

NEGATIVE_PROMPT = (
    "blurry, out of focus, low resolution, low detail, smeared texture, motion blur, "
    "deformed object, distorted shape, broken geometry, duplicate object, multiple subjects, "
    "bad composition, cropped subject, washed out image, same viewpoint as the input image, "
    "same camera angle as the input image"
)


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


def parse_item_idx_from_ref_gt_crop(ref_gt_crop_path: str) -> int | None:
    stem = Path(ref_gt_crop_path).stem
    prefix = "ref_gt_crop_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    try:
        return int(suffix)
    except ValueError:
        return None


def select_subject_concepts(change_concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    additions = [concept for concept in change_concepts if not concept.get("is_remove", False)]
    removals = [concept for concept in change_concepts if concept.get("is_remove", False)]
    if additions:
        return additions, "target_subject"
    if removals:
        return removals, "source_subject"
    return [], None


def canonicalize_subject_phrase(text: str) -> str:
    normalized = text.strip().lower()
    for prefix in ("a ", "an ", "the ", "several ", "two ", "three ", "four "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if normalized.endswith("ies") and len(normalized) > 3:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("s") and not normalized.endswith("ss") and len(normalized) > 3:
        normalized = normalized[:-1]
    return normalized


def summarize_subject_from_concepts(change_concepts: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    chosen_concepts, subject_role = select_subject_concepts(change_concepts)
    if not chosen_concepts:
        return None, subject_role

    sub_prompts = []
    seen_canonical = set()
    for concept in chosen_concepts:
        for sub_prompt in concept.get("sub_prompts_generated", []):
            normalized = sub_prompt.strip()
            canonical = canonicalize_subject_phrase(normalized)
            if normalized and canonical not in seen_canonical:
                sub_prompts.append(normalized)
                seen_canonical.add(canonical)
    if sub_prompts:
        return " and ".join(sub_prompts), subject_role

    descriptions = []
    for concept in chosen_concepts:
        total_prompt = concept.get("total_prompt")
        if total_prompt:
            normalized = total_prompt.strip()
            if normalized not in descriptions:
                descriptions.append(normalized)
    if descriptions:
        return " and ".join(descriptions), subject_role

    return None, subject_role


def load_subject_metadata(processing_log_path: Path, item_idx: int | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "item_idx": item_idx,
        "subject_description": None,
        "subject_role": None,
        "change_concepts": [],
        "instruction": None,
    }
    if item_idx is None or not processing_log_path.exists():
        return metadata

    with processing_log_path.open() as f:
        processing_log = json.load(f)

    matched = next((item for item in processing_log if item.get("item_idx") == item_idx), None)
    if matched is None:
        return metadata

    change_concepts = matched.get("change_concepts", [])
    subject_description, subject_role = summarize_subject_from_concepts(change_concepts)
    metadata.update(
        {
            "subject_description": subject_description,
            "subject_role": subject_role,
            "change_concepts": change_concepts,
            "instruction": matched.get("instruction"),
        }
    )
    return metadata


def select_backgrounds(rng: random.Random, num_backgrounds: int):
    categories = []
    for item in PROMPT_BANK:
        if item["category"] not in categories:
            categories.append(item["category"])

    chosen = []
    shuffled_categories = categories[:]
    rng.shuffle(shuffled_categories)
    for category in shuffled_categories:
        options = [item for item in PROMPT_BANK if item["category"] == category]
        chosen.append(rng.choice(options))
        if len(chosen) >= min(num_backgrounds, len(categories)):
            break

    if num_backgrounds > len(chosen):
        remaining = [item for item in PROMPT_BANK if item not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: num_backgrounds - len(chosen)])

    return chosen[:num_backgrounds]


def select_viewpoints(rng: random.Random, num_viewpoints: int):
    viewpoints = VIEWPOINT_BANK[:]
    rng.shuffle(viewpoints)
    return viewpoints[: min(num_viewpoints, len(viewpoints))]


def preprocess_image(image: Image.Image, target_size: int = 1024) -> tuple[Image.Image, int, int]:
    contained = ImageOps.contain(image, (target_size, target_size), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - contained.width) // 2
    offset_y = (target_size - contained.height) // 2
    canvas.paste(contained, (offset_x, offset_y))
    return canvas, target_size, target_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/home/yanai-lab/xiong-p/qwen/dataset_qwen_pe_test.json")
    parser.add_argument("--image_base", default="/home/yanai-lab/xiong-p/qwen/picobanana/openimages")
    parser.add_argument(
        "--processing_log_path",
        default="/home/yanai-lab/xiong-p/qwen/picobanana/openimages/pico_sam_output_ALL_20251206_032609/processing_log.json",
    )
    parser.add_argument("--output_dir", default="/home/yanai-lab/xiong-p/qwen/outputs/subject_driven_diffusers_cuda2")
    parser.add_argument("--sample_seed", type=int, default=20260522)
    parser.add_argument("--subject_idx", type=int, default=None)
    parser.add_argument("--num_backgrounds", type=int, default=6)
    parser.add_argument("--num_viewpoints", type=int, default=1)
    parser.add_argument("--pair_backgrounds_with_viewpoints", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260522, 20260523])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--task_rank", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--subject_description_override", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    image_base = Path(args.image_base)
    processing_log_path = Path(args.processing_log_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with dataset_path.open() as f:
        data = json.load(f)

    rng = random.Random(args.sample_seed)
    subject_idx = args.subject_idx if args.subject_idx is not None else rng.randrange(len(data))
    sample = data[subject_idx]
    input_image_path = image_base / sample["ref_gt_crop"]
    item_idx = parse_item_idx_from_ref_gt_crop(sample["ref_gt_crop"])
    subject_metadata = load_subject_metadata(processing_log_path, item_idx)
    original_subject_description = subject_metadata["subject_description"]
    if args.subject_description_override:
        subject_metadata["subject_description"] = args.subject_description_override
        subject_metadata["subject_role"] = "qwen_cleaned_override"
    backgrounds = select_backgrounds(rng, args.num_backgrounds)
    viewpoints = select_viewpoints(rng, args.num_viewpoints)

    if args.pair_backgrounds_with_viewpoints and len(viewpoints) < len(backgrounds):
        raise ValueError("Need at least as many viewpoints as backgrounds when pairing them uniquely.")

    print(f"subject_idx = {subject_idx}", flush=True)
    print(f"item_idx = {item_idx}", flush=True)
    print(f"input_image = {input_image_path}", flush=True)
    print(f"subject_description = {subject_metadata['subject_description']}", flush=True)
    print(f"subject_role = {subject_metadata['subject_role']}", flush=True)
    print(f"background_ids = {[item['background_prompt_id'] for item in backgrounds]}", flush=True)
    print(f"viewpoint_ids = {[item['viewpoint_id'] for item in viewpoints]}", flush=True)
    print(f"seeds = {args.seeds}", flush=True)
    print(f"task_rank = {args.task_rank}/{args.num_tasks}", flush=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    input_image = load_image(str(input_image_path)).convert("RGB")
    input_image, resized_width, resized_height = preprocess_image(input_image)

    if args.pair_backgrounds_with_viewpoints:
        bg_viewpoint_pairs = list(zip(backgrounds, viewpoints))
    else:
        bg_viewpoint_pairs = [(bg, viewpoint) for bg in backgrounds for viewpoint in viewpoints]

    jobs = []
    for bg, viewpoint in bg_viewpoint_pairs:
        prompt = build_prompt(
            bg["scene_background"],
            bg["target_scene"],
            subject_metadata["subject_description"],
            viewpoint["camera_instruction"],
        )
        for seed in args.seeds:
            jobs.append(
                {
                    "background": bg,
                    "viewpoint": viewpoint,
                    "prompt": prompt,
                    "seed": seed,
                }
            )

    jobs = jobs[args.task_rank :: args.num_tasks]
    print(f"assigned_jobs = {len(jobs)}", flush=True)

    for job in jobs:
        bg = job["background"]
        viewpoint = job["viewpoint"]
        prompt = job["prompt"]
        seed = job["seed"]
        stem = f"sample_{subject_idx}_{bg['background_prompt_id']}_{viewpoint['viewpoint_id']}_seed{seed}"
        image_path = output_dir / f"{stem}.png"
        meta_path = output_dir / f"{stem}.json"
        if image_path.exists() and meta_path.exists():
            print(f"skipping existing = {image_path}", flush=True)
            continue

        print(
            f"running background={bg['background_prompt_id']} viewpoint={viewpoint['viewpoint_id']} seed={seed}",
            flush=True,
        )
        result = pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=input_image.width,
            height=input_image.height,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        image = result.images[0]
        if image.size != (resized_width, resized_height):
            image = image.crop((0, 0, resized_width, resized_height))
        image.save(image_path)
        meta_path.write_text(
            json.dumps(
                {
                    "subject_idx": subject_idx,
                    "item_idx": item_idx,
                    "input_image_path": str(input_image_path),
                    "source_image_path": str(image_base / sample["image"]),
                    "edit_image_path": str(image_base / sample["edit_image"][0]),
                    "ref_gt_path": str(image_base / sample["ref_gt"]),
                    "ref_gt_crop_path": str(input_image_path),
                    "original_subject_description": original_subject_description,
                    "subject_description": subject_metadata["subject_description"],
                    "subject_role": subject_metadata["subject_role"],
                    "instruction": subject_metadata["instruction"],
                    "change_concepts": subject_metadata["change_concepts"],
                    "background_prompt_id": bg["background_prompt_id"],
                    "scene_background": bg["scene_background"],
                    "target_scene": bg["target_scene"],
                    "viewpoint_id": viewpoint["viewpoint_id"],
                    "camera_instruction": viewpoint["camera_instruction"],
                    "prompt": prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "seed": seed,
                    "steps": args.steps,
                    "model_id": args.model_id,
                    "processed_width": resized_width,
                    "processed_height": resized_height,
                    "task_rank": args.task_rank,
                    "num_tasks": args.num_tasks,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"saved_image = {image_path}", flush=True)


if __name__ == "__main__":
    main()
