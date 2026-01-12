import cv2
import numpy as np
import os
from PIL import Image

BASE_DIR = "pico-banana-400k-subject_driven"
MAX_PIXELS = 1048576 

def guided_filter(I, p, r, eps):
    """Guided Filter implementation"""
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * I + mean_b
    return q

def _compute_target_size(width, height, max_pixels=MAX_PIXELS, div=16):
    if width * height > max_pixels:
        scale = (width * height / max_pixels) ** 0.5
        height = int(height / scale)
        width = int(width / scale)
    height = (height // div) * div
    width = (width // div) * div
    return width, height

def run_test(item_idx, before_rel, after_rel, mask_rel, output_subdir):
    print(f"--- Running High Dist Test for Item {item_idx} ({output_subdir}) ---")
    
    before_path = os.path.join(BASE_DIR, before_rel)
    after_path = os.path.join(BASE_DIR, after_rel)
    mask_path = os.path.join(BASE_DIR, mask_rel)
    output_dir = os.path.join("tmp", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    if not (os.path.exists(before_path) and os.path.exists(after_path) and os.path.exists(mask_path)):
        print(f"❌ Error: Missing files for {item_idx}")
        return

    # 1. Load & Resize
    img_before_pil = Image.open(before_path).convert("RGB")
    img_after_pil = Image.open(after_path).convert("RGB")
    img_mask_pil = Image.open(mask_path).convert("L")

    w, h = img_after_pil.size
    tw, th = _compute_target_size(w, h)
    img_before = img_before_pil.resize((tw, th), Image.Resampling.LANCZOS)
    img_after = img_after_pil.resize((tw, th), Image.Resampling.LANCZOS)
    img_mask = img_mask_pil.resize((tw, th), Image.Resampling.NEAREST)

    # 2. Trimap logic (Iter=9)
    input_mask = np.array(img_mask).astype(np.float64) / 255.0 
    kernel = np.ones((5, 5), np.uint8)
    input_mask_u8 = (input_mask * 255).astype(np.uint8)

    mask_inner_u8 = cv2.dilate(input_mask_u8, kernel, iterations=9)
    mask_outer_u8 = cv2.erode(input_mask_u8, kernel, iterations=9)
    
    p = input_mask.copy()
    p[mask_outer_u8 > 127] = 1.0 
    p[mask_inner_u8 < 127] = 0.0 
    
    # 3. Dual-Guided Filter (r=16, eps=1e-2)
    guide_b = np.array(img_before.convert("L")).astype(np.float64) / 255.0
    guide_a = np.array(img_after.convert("L")).astype(np.float64) / 255.0
    
    refined_b = guided_filter(guide_b, p, r=16, eps=1e-3)
    refined_a = guided_filter(guide_a, p, r=16, eps=1e-3)
    
    # MAX logic
    refined_mask = np.maximum(refined_b, refined_a)
    refined_mask = np.clip(refined_mask, 0, 1)

    # 4. Composite
    final_mask_3d = np.expand_dims(refined_mask, axis=2)
    arr_b = np.array(img_before).astype(float)
    arr_a = np.array(img_after).astype(float)
    arr_final = final_mask_3d * arr_b + (1.0 - final_mask_3d) * arr_a
    res_u8 = np.clip(arr_final, 0, 255).astype(np.uint8)
    
    # 5. Save Results
    cv2.imwrite(os.path.join(output_dir, "final_fixed_image.png"), cv2.cvtColor(res_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "mask_processed.png"), (refined_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "debug_inner.png"), mask_inner_u8)
    cv2.imwrite(os.path.join(output_dir, "debug_outer.png"), mask_outer_u8)
    cv2.imwrite(os.path.join(output_dir, "refined_before.png"), (np.clip(refined_b, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "refined_after.png"), (np.clip(refined_a, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "before_resized.png"), cv2.cvtColor(np.array(img_before), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "after_resized.png"), cv2.cvtColor(np.array(img_after), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "original_mask.png"), np.array(img_mask))
    
    print(f"✅ Results for {item_idx} saved to {output_dir}/")

def main():
    # 1. Add Task (10010)
    run_test(10010, 
             "openimages/source/train_0/04b5dc69a50e90bf.jpg",
             "openimages/edited/sft/62685.png",
             "openimages/dreamsim_mask_audit_mask/item_10010_bg_check_mask.png",
             "add_test")

    # 2. Remove Task (10058)
    run_test(10058, 
             "openimages/source/train_0/04d06816e1ed6cbd.jpg",
             "openimages/edited/sft/63393.png",
             "openimages/dreamsim_mask_audit_mask/item_10058_bg_check_mask.png",
             "remove_test")

    # 3. Replace Task (10076)
    run_test(10076, 
             "openimages/source/train_0/04d22f38c104dc59.jpg",
             "openimages/edited/sft/63470.png",
             "openimages/dreamsim_mask_audit_mask/item_10076_bg_check_mask.png",
             "replace_test")

if __name__ == "__main__":
    main()
