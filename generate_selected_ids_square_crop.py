import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


DATASET_JSON = Path("dataset_qwen_pe_fixed.json")
OPENIMAGES_ROOT = Path("pico-banana-400k-subject_driven/openimages")
MODEL_DIR = Path("pico_test/qwen_results_top1000")
OUTPUT_DIR = Path("outputs/qwen_selected_ids_mask_squarecrop")
MASK_ALPHA = 0.5
MAX_WORKERS = 16

SELECTED_IDS = [
    407, 727, 740, 868, 1252, 3270, 4426, 5509, 5877, 5881,
    6496, 8131, 16334, 16466, 20399, 24621, 25295, 25473, 27742, 28093,
    30718, 31857, 33052, 33536, 33555, 33676, 33947, 34240, 34408, 34523,
    36432, 38443, 40402, 40716, 42005, 42272, 42396, 42711, 43108,
]


def load_rgb_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def make_placeholder(size: tuple[int, int], text: str = "model image missing") -> Image.Image:
    img = Image.new("RGB", size, (24, 24, 24))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(230, 230, 230))
    return img


def trim_black_padding(image: Image.Image, threshold: int = 8, min_black_ratio: float = 0.98) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    h, w, _ = arr.shape
    near_black = np.all(arr <= threshold, axis=2)

    def find_top() -> int:
        top = 0
        while top < h and near_black[top].mean() >= min_black_ratio:
            top += 1
        return top

    def find_bottom() -> int:
        bottom = h - 1
        while bottom >= 0 and near_black[bottom].mean() >= min_black_ratio:
            bottom -= 1
        return bottom

    def find_left() -> int:
        left = 0
        while left < w and near_black[:, left].mean() >= min_black_ratio:
            left += 1
        return left

    def find_right() -> int:
        right = w - 1
        while right >= 0 and near_black[:, right].mean() >= min_black_ratio:
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


def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    image_np = np.array(image)
    mask_np = np.array(mask.convert("L"))

    mask_colored = np.zeros_like(image_np)
    mask_colored[:, :, 0] = mask_np

    mask_binary = (mask_np > 0).astype(float)
    mask_binary = np.expand_dims(mask_binary, axis=2)

    overlayed = image_np * (1 - alpha * mask_binary) + mask_colored * alpha * mask_binary
    return Image.fromarray(overlayed.astype(np.uint8))


def compute_mask_square_crop(mask: Image.Image) -> tuple[int, int, int, int]:
    mask_np = np.array(mask.convert("L"))
    h, w = mask_np.shape
    ys, xs = np.where(mask_np > 0)

    side = min(w, h)

    if len(xs) == 0 or len(ys) == 0:
        left = (w - side) // 2
        top = (h - side) // 2
        return left, top, left + side, top + side

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    x_low = max(0, x_max - side + 1)
    x_high = min(x_min, w - side)
    y_low = max(0, y_max - side + 1)
    y_high = min(y_min, h - side)

    bbox_center_x = (x_min + x_max) / 2.0
    bbox_center_y = (y_min + y_max) / 2.0

    ideal_left = int(round(bbox_center_x - side / 2))
    ideal_top = int(round(bbox_center_y - side / 2))

    if x_low <= x_high:
        left = min(max(ideal_left, x_low), x_high)
    else:
        left = min(max(ideal_left, 0), max(0, w - side))

    if y_low <= y_high:
        top = min(max(ideal_top, y_low), y_high)
    else:
        top = min(max(ideal_top, 0), max(0, h - side))

    return left, top, left + side, top + side


def compose_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    if right.size != left.size:
        right = right.resize(left.size, Image.LANCZOS)
    canvas = Image.new("RGB", (left.width * 2, left.height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def parse_fixed_id(sample: dict) -> int | None:
    edit_images = sample.get("edit_image", [])
    if not edit_images:
        return None
    name = Path(edit_images[0]).stem
    try:
        return int(name.split("_")[-1])
    except Exception:
        return None


def process_one(sample_id: int, sample: dict) -> tuple[str, str | None]:
    edit_images = sample.get("edit_image", [])
    if not edit_images:
        return "skipped", f"[WARN] id={sample_id}: no edit_image"

    input_rel = edit_images[0]
    mask_rel = sample.get("back_mask")
    input_path = OPENIMAGES_ROOT / input_rel
    mask_path = OPENIMAGES_ROOT / mask_rel if mask_rel else None

    if not input_path.exists():
        return "skipped", f"[WARN] id={sample_id}: input missing {input_path}"

    if not mask_path or not mask_path.exists():
        return "skipped", f"[WARN] id={sample_id}: mask missing {mask_path}"

    file_name = Path(input_rel).name
    model_path = MODEL_DIR / file_name

    try:
        input_img = load_rgb_image(input_path)
        mask_img = Image.open(mask_path).convert("L")
        if mask_img.size != input_img.size:
            mask_img = mask_img.resize(input_img.size, Image.NEAREST)

        left = overlay_mask_on_image(input_img, mask_img, alpha=MASK_ALPHA)

        if model_path.exists():
            right = load_rgb_image(model_path)
            right = trim_black_padding(right)
            if right.size != input_img.size:
                right = right.resize(input_img.size, Image.LANCZOS)
        else:
            right = make_placeholder(input_img.size)

        crop_box = compute_mask_square_crop(mask_img)
        left_crop = left.crop(crop_box)
        right_crop = right.crop(crop_box)

        merged = compose_side_by_side(left_crop, right_crop)
        out_name = f"fixed_{sample_id}_inputmask_vs_model_squarecrop.png"
        merged.save(OUTPUT_DIR / out_name)
        return "saved", None
    except Exception as e:
        return "skipped", f"[WARN] id={sample_id}: {e}"


def main() -> None:
    if not DATASET_JSON.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_JSON}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    with open(DATASET_JSON, "r") as f:
        dataset = json.load(f)

    id_to_sample = {}
    for sample in dataset:
        sample_id = parse_fixed_id(sample)
        if sample_id is not None and sample_id not in id_to_sample:
            id_to_sample[sample_id] = sample

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [sample_id for sample_id in SELECTED_IDS if sample_id in id_to_sample]
    missing_in_dataset = [sample_id for sample_id in SELECTED_IDS if sample_id not in id_to_sample]

    saved = 0
    skipped = 0
    warnings: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, sample_id, id_to_sample[sample_id]) for sample_id in targets]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping selected IDs", unit="img"):
            status, warn = future.result()
            if status == "saved":
                saved += 1
            else:
                skipped += 1
            if warn:
                warnings.append(warn)

    print("=" * 80)
    print(f"selected_ids={len(SELECTED_IDS)}, found_in_dataset={len(targets)}")
    print(f"saved={saved}, skipped={skipped}")
    print(f"output_dir={OUTPUT_DIR}")
    if missing_in_dataset:
        print(f"missing_in_dataset={len(missing_in_dataset)}: {missing_in_dataset}")
    if warnings:
        print(f"warnings={len(warnings)} (showing first 20)")
        for line in warnings[:20]:
            print(line)


if __name__ == "__main__":
    main()
