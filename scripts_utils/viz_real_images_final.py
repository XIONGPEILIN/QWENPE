import json
import os
import random
import cv2
import pandas as pd
import numpy as np

# Configuration
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
RESULTS_DIR = 'final_comparison_results'
BASE_DIR = 'pico-banana-400k-subject_driven/openimages'
OUTPUT_IMAGE = 'visualize_acc_clothing_final.jpg'
NUM_SAMPLES_PER_TYPE = 2
CELL_SIZE = 512

MODEL_DIRS = {
    'qwen_w_ste': 'pico_test/qwen_results_top1000',
    'ace_plus': 'pico_test/ace_plus_results_top1000',
    'flux': 'pico_test/flux_results_top1000'
}

TARGET_TYPES = {
    "ACCESSORY": "Add/Remove/Replace Accessories (glasses, hats, jewelry, masks)",
    "CLOTHING": "Clothing edit (change color/outfit)"
}

def draw_text_with_bg(img, text, x, y, font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(0, min(x, img.shape[1] - w))
    y = max(h + 5, min(y, img.shape[0] - 5))
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def get_scores(df, filename):
    row = df[df['filename'] == filename]
    return row.iloc[0] if len(row) > 0 else None

def visualize_final():
    with open(TOP1000_JSON, 'r') as f:
        dataset = json.load(f)
    
    dfs = {m: pd.read_csv(os.path.join(RESULTS_DIR, f"{m}_full.csv")) 
           for m in MODEL_DIRS.keys() if os.path.exists(os.path.join(RESULTS_DIR, f"{m}_full.csv"))}

    samples = []
    for short_type, full_type in TARGET_TYPES.items():
        candidates = [item for item in dataset if item.get('edit_type') == full_type]
        # Use basename of edit_image (e.g., fixed_10004.png) to find model results
        valid = [c for c in candidates if all(os.path.exists(os.path.join(d, os.path.basename(c['edit_image'][0]))) for d in MODEL_DIRS.values())]
        if valid:
            samples.extend([(short_type, item) for item in random.sample(valid, min(NUM_SAMPLES_PER_TYPE, len(valid)))])

    if not samples:
        print("Still no samples found! Please check file availability.")
        return

    cols = 3 + len(MODEL_DIRS)
    canvas = np.zeros((len(samples) * CELL_SIZE, cols * CELL_SIZE, 3), dtype=np.uint8)

    for i, (cat, item) in enumerate(samples):
        y_off = i * CELL_SIZE
        source_rel = item['edit_image'][0]
        gt_rel = item['image']
        fname = os.path.basename(source_rel) # Correct: Model results use source filenames (fixed_*.png)
        
        # --- Col 0: ORIGINAL (Source) ---
        src_img = cv2.imread(os.path.join(BASE_DIR, source_rel))
        if src_img is not None:
            src_img = cv2.resize(src_img, (CELL_SIZE, CELL_SIZE))
            draw_text_with_bg(src_img, "ORIGINAL (Source)", 10, 30, color=(0, 255, 255), thickness=2)
            prompt = item['prompt'].replace("Picture 1 is the image to modify. ", "")
            draw_text_with_bg(src_img, prompt[:40]+"...", 10, 60, font_scale=0.4)
            canvas[y_off:y_off+CELL_SIZE, 0:CELL_SIZE] = src_img

        # --- Col 1: TARGET (GT) ---
        gt_img = cv2.imread(os.path.join(BASE_DIR, gt_rel))
        if gt_img is not None:
            gt_img = cv2.resize(gt_img, (CELL_SIZE, CELL_SIZE))
            draw_text_with_bg(gt_img, "TARGET (GT)", 10, 30, color=(0, 255, 0), thickness=2)
            canvas[y_off:y_off+CELL_SIZE, CELL_SIZE:2*CELL_SIZE] = gt_img

        # --- Col 2: Mask ---
        mask_img = cv2.imread(os.path.join(BASE_DIR, item['back_mask']))
        bbox = None
        if mask_img is not None:
            gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(gray)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                bbox = (int(x*CELL_SIZE/mask_img.shape[1]), int(y*CELL_SIZE/mask_img.shape[0]), 
                        int(w*CELL_SIZE/mask_img.shape[1]), int(h*CELL_SIZE/mask_img.shape[0]))
            mask_vis = cv2.applyColorMap(cv2.resize(mask_img, (CELL_SIZE, CELL_SIZE)), cv2.COLORMAP_JET)
            draw_text_with_bg(mask_vis, "Edit Mask", 10, 30)
            canvas[y_off:y_off+CELL_SIZE, 2*CELL_SIZE:3*CELL_SIZE] = mask_vis

        # --- Col 3+: Models ---
        for j, (m_name, m_dir) in enumerate(MODEL_DIRS.items()):
            x_off = (3 + j) * CELL_SIZE
            res_img = cv2.imread(os.path.join(m_dir, fname))
            if res_img is not None:
                res_img = cv2.resize(res_img, (CELL_SIZE, CELL_SIZE))
                if m_name in dfs:
                    row = get_scores(dfs[m_name], fname)
                    if row is not None:
                        draw_text_with_bg(res_img, m_name.upper(), 10, 30, color=(0, 255, 0), font_scale=0.7, thickness=2)
                        draw_text_with_bg(res_img, f"Mask: S={row['siglip2_i_mask']:.2f} D={row['dreamsim_mask']:.2f}", 10, 60, font_scale=0.5)
                        if bbox:
                            bx, by, bw, bh = bbox
                            cv2.rectangle(res_img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
                canvas[y_off:y_off+CELL_SIZE, x_off:x_off+CELL_SIZE] = res_img

    cv2.imwrite(OUTPUT_IMAGE, canvas)
    print(f"Final visualization saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    visualize_final()
