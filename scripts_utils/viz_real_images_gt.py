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
OUTPUT_IMAGE = 'visualize_acc_clothing_with_gt.jpg'
NUM_SAMPLES_PER_TYPE = 2  # 2 Acc + 2 Cloth
CELL_SIZE = 512

# Model mapping: Name in CSV -> Directory path
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
    if len(row) > 0:
        return row.iloc[0]
    return None

def visualize_with_gt():
    # 1. Load Dataset
    with open(TOP1000_JSON, 'r') as f:
        dataset = json.load(f)
    
    # 2. Load Scores Dataframes
    dfs = {}
    for m in MODEL_DIRS.keys():
        path = os.path.join(RESULTS_DIR, f"{m}_full.csv")
        if os.path.exists(path):
            dfs[m] = pd.read_csv(path)

    # 3. Select Samples
    samples = []
    for short_type, full_type in TARGET_TYPES.items():
        candidates = [item for item in dataset if item.get('edit_type') == full_type]
        valid_candidates = []
        for cand in candidates:
            fname = os.path.basename(cand['edit_image'][0])
            # Check models AND Ref GT
            if all(os.path.exists(os.path.join(d, fname)) for d in MODEL_DIRS.values()) and \
               os.path.exists(os.path.join(BASE_DIR, cand['ref_gt'])):
                valid_candidates.append(cand)
        
        if valid_candidates:
            picked = random.sample(valid_candidates, min(NUM_SAMPLES_PER_TYPE, len(valid_candidates)))
            samples.extend([(short_type, item) for item in picked])

    if not samples:
        print("No valid samples found.")
        return

    # 4. Draw Canvas
    # Cols: Source | Ref GT | Mask | Qwen | ACE+ | Flux
    cols = 3 + len(MODEL_DIRS)
    rows = len(samples)
    canvas = np.zeros((rows * CELL_SIZE, cols * CELL_SIZE, 3), dtype=np.uint8)

    for i, (cat, item) in enumerate(samples):
        y_off = i * CELL_SIZE
        fname = os.path.basename(item['edit_image'][0])
        
        # --- Col 0: Source ---
        src_path = os.path.join(BASE_DIR, item['image'])
        src_img = cv2.imread(src_path)
        if src_img is not None:
            src_img = cv2.resize(src_img, (CELL_SIZE, CELL_SIZE))
            prompt_lines = [item['prompt'][j:j+40] for j in range(0, len(item['prompt']), 40)]
            for k, line in enumerate(prompt_lines[:3]):
                draw_text_with_bg(src_img, line, 10, 30 + k*25, font_scale=0.5)
            draw_text_with_bg(src_img, f"TYPE: {cat}", 10, src_img.shape[0]-20, color=(0,255,255))
            canvas[y_off:y_off+CELL_SIZE, 0:CELL_SIZE] = src_img

        # --- Col 1: Reference GT ---
        gt_path = os.path.join(BASE_DIR, item['ref_gt'])
        gt_img = cv2.imread(gt_path)
        if gt_img is not None:
            gt_img = cv2.resize(gt_img, (CELL_SIZE, CELL_SIZE))
            draw_text_with_bg(gt_img, "Reference GT", 10, 30, color=(0, 255, 0), thickness=2)
            canvas[y_off:y_off+CELL_SIZE, CELL_SIZE:2*CELL_SIZE] = gt_img
        else:
            # Fallback black
            pass

        # --- Col 2: Mask + BBox ---
        mask_path = os.path.join(BASE_DIR, item['back_mask'])
        mask_img = cv2.imread(mask_path)
        bbox = None
        if mask_img is not None:
            gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(gray)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                sx = CELL_SIZE / mask_img.shape[1]
                sy = CELL_SIZE / mask_img.shape[0]
                bbox = (int(x*sx), int(y*sy), int(w*sx), int(h*sy))
            
            mask_img = cv2.resize(mask_img, (CELL_SIZE, CELL_SIZE))
            mask_vis = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
            draw_text_with_bg(mask_vis, "Edit Mask", 10, 30)
            if bbox:
                bx, by, bw, bh = bbox
                cv2.rectangle(mask_vis, (bx, by), (bx+bw, by+bh), (255, 255, 255), 2)
            canvas[y_off:y_off+CELL_SIZE, 2*CELL_SIZE:3*CELL_SIZE] = mask_vis

        # --- Col 3+: Models ---
        for j, (m_name, m_dir) in enumerate(MODEL_DIRS.items()):
            x_off = (3 + j) * CELL_SIZE
            img_path = os.path.join(m_dir, fname)
            res_img = cv2.imread(img_path)
            
            if res_img is not None:
                res_img = cv2.resize(res_img, (CELL_SIZE, CELL_SIZE))
                
                if m_name in dfs:
                    row = get_scores(dfs[m_name], fname)
                    if row is not None:
                        s_mask = row.get('siglip2_i_mask', 0)
                        d_mask = row.get('dreamsim_mask', 0)
                        draw_text_with_bg(res_img, m_name.upper(), 10, 30, color=(0, 255, 0), font_scale=0.7, thickness=2)
                        draw_text_with_bg(res_img, f"Mask: S={s_mask:.2f} D={d_mask:.2f}", 10, 60, font_scale=0.6)
                        
                        if bbox:
                            bx, by, bw, bh = bbox
                            cv2.rectangle(res_img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
                            s_bbox = row.get('siglip2_i_bbox', 0)
                            d_bbox = row.get('dreamsim_bbox', 0)
                            cx = bx + 5
                            cy = by + bh + 20
                            if cy > CELL_SIZE - 20: cy = by - 10
                            draw_text_with_bg(res_img, f"BBox S={s_bbox:.2f} D={d_bbox:.2f}", cx, cy, font_scale=0.5, bg_color=(50, 50, 50))

                canvas[y_off:y_off+CELL_SIZE, x_off:x_off+CELL_SIZE] = res_img

    cv2.imwrite(OUTPUT_IMAGE, canvas)
    print(f"Saved: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    visualize_with_gt()
