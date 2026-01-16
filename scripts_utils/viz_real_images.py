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
OUTPUT_IMAGE = 'visualize_acc_clothing_real_images.jpg'
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
    # Ensure we don't draw outside
    x = max(0, min(x, img.shape[1] - w))
    y = max(h + 5, min(y, img.shape[0] - 5))
    
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def get_scores(df, filename):
    row = df[df['filename'] == filename]
    if len(row) > 0:
        return row.iloc[0]
    return None

def visualize_real_images():
    # 1. Load Dataset
    with open(TOP1000_JSON, 'r') as f:
        dataset = json.load(f)
    
    # 2. Load Scores Dataframes
    dfs = {}
    for m in MODEL_DIRS.keys():
        path = os.path.join(RESULTS_DIR, f"{m}_full.csv")
        if os.path.exists(path):
            dfs[m] = pd.read_csv(path)
        else:
            print(f"Warning: Score file {path} not found.")

    # 3. Select Samples
    samples = []
    for short_type, full_type in TARGET_TYPES.items():
        candidates = [item for item in dataset if item.get('edit_type') == full_type]
        # Filter for existing files in ALL model dirs
        valid_candidates = []
        for cand in candidates:
            fname = os.path.basename(cand['edit_image'][0])
            if all(os.path.exists(os.path.join(d, fname)) for d in MODEL_DIRS.values()):
                valid_candidates.append(cand)
        
        if valid_candidates:
            picked = random.sample(valid_candidates, min(NUM_SAMPLES_PER_TYPE, len(valid_candidates)))
            samples.extend([(short_type, item) for item in picked])

    if not samples:
        print("No valid samples found that exist in all model directories.")
        return

    # 4. Draw Canvas
    # Layout:
    # Col 0: Source + Prompt
    # Col 1: Mask
    # Col 2: Qwen
    # Col 3: ACE+
    # Col 4: Flux
    
    cols = 2 + len(MODEL_DIRS)
    rows = len(samples)
    canvas_h = rows * CELL_SIZE
    canvas_w = cols * CELL_SIZE
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, (cat, item) in enumerate(samples):
        y_off = i * CELL_SIZE
        fname = os.path.basename(item['edit_image'][0])
        
        # --- Col 0: Source ---
        src_path = os.path.join(BASE_DIR, item['image'])
        src_img = cv2.imread(src_path)
        if src_img is not None:
            src_img = cv2.resize(src_img, (CELL_SIZE, CELL_SIZE))
            # Draw Prompt
            prompt_lines = [item['prompt'][j:j+40] for j in range(0, len(item['prompt']), 40)]
            for k, line in enumerate(prompt_lines[:3]): # Max 3 lines
                draw_text_with_bg(src_img, line, 10, 30 + k*25, font_scale=0.5)
            
            draw_text_with_bg(src_img, f"TYPE: {cat}", 10, src_img.shape[0]-20, color=(0,255,255))
            canvas[y_off:y_off+CELL_SIZE, 0:CELL_SIZE] = src_img

        # --- Col 1: Mask + BBox Info ---
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
            canvas[y_off:y_off+CELL_SIZE, CELL_SIZE:2*CELL_SIZE] = mask_vis

        # --- Col 2+: Models ---
        for j, (m_name, m_dir) in enumerate(MODEL_DIRS.items()):
            x_off = (2 + j) * CELL_SIZE
            
            img_path = os.path.join(m_dir, fname)
            res_img = cv2.imread(img_path)
            
            if res_img is None:
                res_img = np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
                draw_text_with_bg(res_img, "IMG MISSING", 10, 100, color=(0,0,255))
            else:
                res_img = cv2.resize(res_img, (CELL_SIZE, CELL_SIZE))
            
            # Scores
            if m_name in dfs:
                row = get_scores(dfs[m_name], fname)
                if row is not None:
                    # Top: SigLIP & DreamSim (Mask)
                    s_mask = row.get('siglip2_i_mask', 0)
                    d_mask = row.get('dreamsim_mask', 0)
                    score_txt = f"Mask: S={s_mask:.3f} D={d_mask:.3f}"
                    draw_text_with_bg(res_img, m_name.upper(), 10, 30, color=(0, 255, 0), font_scale=0.7, thickness=2)
                    draw_text_with_bg(res_img, score_txt, 10, 60, font_scale=0.6)
                    
                    # BBox Drawing & Scores
                    if bbox:
                        bx, by, bw, bh = bbox
                        cv2.rectangle(res_img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
                        
                        s_bbox = row.get('siglip2_i_bbox', 0)
                        d_bbox = row.get('dreamsim_bbox', 0)
                        
                        # Put BBox scores near the box (try center, fallback to bottom if too small)
                        # Ensure we don't cover the object completely if small
                        cx = bx + 5
                        cy = by + bh + 20
                        if cy > CELL_SIZE - 20: cy = by - 10
                        
                        bbox_txt = f"BBox S={s_bbox:.3f} D={d_bbox:.3f}"
                        draw_text_with_bg(res_img, bbox_txt, cx, cy, font_scale=0.5, bg_color=(50, 50, 50))

            canvas[y_off:y_off+CELL_SIZE, x_off:x_off+CELL_SIZE] = res_img

    cv2.imwrite(OUTPUT_IMAGE, canvas)
    print(f"Comparison saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    visualize_real_images()
