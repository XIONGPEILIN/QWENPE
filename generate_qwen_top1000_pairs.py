import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


DATASET_JSON = Path("dataset_qwen_pe_top1000_captioned.json")
OPENIMAGES_ROOT = Path("pico-banana-400k-subject_driven/openimages")
MODEL_DIR = Path("pico_test/qwen_results_top1000")
OUTPUT_DIR = Path("outputs/qwen_top1000_inputmask_vs_model")
MASK_ALPHA = 0.5
MAX_WORKERS = min(32, max(4, (os.cpu_count() or 4) * 2))


def load_rgb_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def trim_black_padding(image: Image.Image, threshold: int = 8, min_black_ratio: float = 0.98) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    h, w, _ = arr.shape

    near_black = np.all(arr <= threshold, axis=2)

    def find_top() -> int:
        top = 0
        while top < h:
            if near_black[top].mean() < min_black_ratio:
                break
            top += 1
        return top

    def find_bottom() -> int:
        bottom = h - 1
        while bottom >= 0:
            if near_black[bottom].mean() < min_black_ratio:
                break
            bottom -= 1
        return bottom

    def find_left() -> int:
        left = 0
        while left < w:
            if near_black[:, left].mean() < min_black_ratio:
                break
            left += 1
        return left

    def find_right() -> int:
        right = w - 1
        while right >= 0:
            if near_black[:, right].mean() < min_black_ratio:
                break
            right -= 1
        return right

    top = find_top()
    bottom = find_bottom()
    left = find_left()
    right = find_right()

    if top >= bottom or left >= right:
        return image

    if top == 0 and left == 0 and bottom == h - 1 and right == w - 1:
        return image

    return image.crop((left, top, right + 1, bottom + 1))


def overlay_mask(image_path: Path, mask_path: Path, alpha: float = 0.5) -> Image.Image:
    image = load_rgb_image(image_path)
    mask = Image.open(mask_path).convert("L")

    if mask.size != image.size:
        mask = mask.resize(image.size, Image.LANCZOS)

    image_np = np.array(image)
    mask_np = np.array(mask)

    mask_colored = np.zeros_like(image_np)
    mask_colored[:, :, 0] = mask_np

    mask_binary = (mask_np > 0).astype(float)
    mask_binary = np.expand_dims(mask_binary, axis=2)

    overlayed = image_np * (1 - alpha * mask_binary) + mask_colored * alpha * mask_binary
    return Image.fromarray(overlayed.astype(np.uint8))


def placeholder(size: tuple[int, int], text: str = "model image missing") -> Image.Image:
    img = Image.new("RGB", size, (24, 24, 24))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(230, 230, 230))
    return img


def compose_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    if right.size != left.size:
        right = right.resize(left.size, Image.LANCZOS)
    canvas = Image.new("RGB", (left.width * 2, left.height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def process_one(idx: int, sample: dict) -> tuple[str, str | None]:
    edit_images = sample.get("edit_image", [])
    if not edit_images:
        return "skipped", None

    input_rel = edit_images[0]
    mask_rel = sample.get("back_mask")
    input_path = OPENIMAGES_ROOT / input_rel
    mask_path = OPENIMAGES_ROOT / mask_rel if mask_rel else None

    if not input_path.exists():
        return "skipped", f"[WARN] idx={idx}, input missing: {input_path}"

    file_name = Path(input_rel).name
    model_path = MODEL_DIR / file_name

    try:
        if mask_path and mask_path.exists():
            left = overlay_mask(input_path, mask_path, alpha=MASK_ALPHA)
        else:
            left = load_rgb_image(input_path)

        if model_path.exists():
            right = load_rgb_image(model_path)
            right = trim_black_padding(right)
        else:
            right = placeholder(left.size)

        merged = compose_side_by_side(left, right)
        out_name = f"{Path(file_name).stem}_inputmask_vs_model.png"
        merged.save(OUTPUT_DIR / out_name)
        return "saved", None
    except Exception as e:
        return "skipped", f"[WARN] idx={idx}, file={file_name}, error={e}"


def main() -> None:
    if not DATASET_JSON.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_JSON}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    with open(DATASET_JSON, "r") as f:
        dataset = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    warnings = []
    total = min(1000, len(dataset))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, idx, sample) for idx, sample in enumerate(dataset[:total])]

        for future in tqdm(as_completed(futures), total=total, desc="Generating pairs", unit="img"):
            status, warn = future.result()
            if status == "saved":
                saved += 1
            else:
                skipped += 1
            if warn:
                warnings.append(warn)

    print("=" * 80)
    print(f"Done. total={total}, saved={saved}, skipped={skipped}")
    print(f"workers={MAX_WORKERS}")
    print(f"model_dir={MODEL_DIR}")
    print(f"output_dir={OUTPUT_DIR}")
    if warnings:
        print(f"warnings={len(warnings)} (showing first 20)")
        for line in warnings[:20]:
            print(line)


if __name__ == "__main__":
    main()
