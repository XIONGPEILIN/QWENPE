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
OUTPUT_IMAGE = 'visualize_acc_clothing_metrics.jpg'
NUM_SAMPLES = 6  # Total samples to show (3 Acc + 3 Cloth)
CELL_SIZE = 384  # Larger size for readability

MODELS = ['qwen_w_ste', 'ace_plus', 'flux'] # Models to compare

TARGET_TYPES = {
    "acc": "Add/Remove/Replace Accessories (glasses, hats, jewelry, masks)",
    "cloth": "Clothing edit (change color/outfit)"
}

def draw_text_with_bg(img, text, x, y, font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(img, (x, y - h - 4), (x + w, y + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

def get_scores(df, filename):
    row = df[df['filename'] == filename]
    if len(row) > 0:
        return {
            'siglip_mask': row.iloc[0].get('siglip2_i_mask', 0),
            'ds_mask': row.iloc[0].get('dreamsim_mask', 0),
            'siglip_bbox': row.iloc[0].get('siglip2_i_bbox', 0),
            'ds_bbox': row.iloc[0].get('dreamsim_bbox', 0)
        }
    return None

def visualize_metrics():
    # 1. Load Dataset
    with open(TOP1000_JSON, 'r') as f:
        dataset = json.load(f)
    
    # 2. Load Scores
    scores = {}
    for m in MODELS:
        path = os.path.join(RESULTS_DIR, f"{m}_full.csv")
        if os.path.exists(path):
            scores[m] = pd.read_csv(path)
    
    # 3. Select Samples
    samples = []
    for short_type, full_type in TARGET_TYPES.items():
        candidates = [item for item in dataset if item.get('edit_type') == full_type]
        if candidates:
            # Pick valid ones that have edit images
            valid = [c for c in candidates if os.path.exists(os.path.join(BASE_DIR, c['edit_image'][0]))]
            picked = random.sample(valid, min(3, len(valid)))
            samples.extend([(short_type, item) for item in picked])

    if not samples:
        print("No matching samples found.")
        return

    # 4. Draw Canvas
    # Cols: Source | GT (if any) | Model 1 | Model 2 | Model 3 ...
    cols = 2 + len(MODELS) 
    rows = len(samples)
    canvas = np.zeros((rows * CELL_SIZE, cols * CELL_SIZE, 3), dtype=np.uint8)

    for i, (cat, item) in enumerate(samples):
        # Row Y
        y_off = i * CELL_SIZE
        
        # --- Col 0: Source ---
        src_path = os.path.join(BASE_DIR, item['image'])
        src_img = cv2.imread(src_path)
        if src_img is not None:
            src_img = cv2.resize(src_img, (CELL_SIZE, CELL_SIZE))
            draw_text_with_bg(src_img, f"Source ({cat})", 10, 30)
            # Wrap prompt
            prompt = item['prompt'][:40] + "..."
            draw_text_with_bg(src_img, prompt, 10, src_img.shape[0]-20, font_scale=0.4)
            canvas[y_off:y_off+CELL_SIZE, 0:CELL_SIZE] = src_img
            
        # --- Col 1: Mask (or Ref GT) ---
        mask_path = os.path.join(BASE_DIR, item['back_mask'])
        mask_img = cv2.imread(mask_path)
        bbox = None
        if mask_img is not None:
            # Find BBox
            gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(gray)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                # Scale bbox to CELL_SIZE
                sx = CELL_SIZE / mask_img.shape[1]
                sy = CELL_SIZE / mask_img.shape[0]
                bbox = (int(x*sx), int(y*sy), int(w*sx), int(h*sy))
            
            mask_img = cv2.resize(mask_img, (CELL_SIZE, CELL_SIZE))
            # Colorize mask
            mask_vis = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
            draw_text_with_bg(mask_vis, "Mask", 10, 30)
            canvas[y_off:y_off+CELL_SIZE, CELL_SIZE:2*CELL_SIZE] = mask_vis

        # --- Col 2+: Models ---
        # Edit filename is needed to lookup scores
        # Note: 'edit_image' in JSON points to 'fixed_images/fixed_XXXX.png'
        # But the CSV might index by filename 'fixed_XXXX.png' or full path
        # Let's check format. Based on previous 'head', CSV uses filename only.
        fname = os.path.basename(item['edit_image'][0])
        
        for j, m_name in enumerate(MODELS):
            x_off = (2 + j) * CELL_SIZE
            
            # Since we don't have the actual generated images from other models locally 
            # (assuming we only have Qwen's or the reference?), 
            # WAIT. If we want to show Flux/ACE+ images, we need their paths.
            # If we don't have them, we can only show scores.
            # Assuming we might NOT have all model images generated in this folder structure.
            # BUT, if we assume we are just visualizing the Ground Truth (Fixed) vs Source, 
            # we can't show other models visually if files aren't there.
            
            # CHECK: Do we have other models' images?
            # Usually 'final_comparison_results' implies we evaluated them.
            # The evaluation script usually takes a folder.
            # If we can't find the image, we just show a black box with scores.
            
            # Let's try to construct a path. Usually eval uses: pred_dir/filename
            # Let's assume standard names or just show scores on a blank slate if missing.
            
            # Placeholder for model image
            model_img = np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
            
            # Lookup Scores
            metrics = None
            if m_name in scores:
                metrics = get_scores(scores[m_name], fname)
            
            if metrics:
                # Text Info
                draw_text_with_bg(model_img, m_name, 10, 30, color=(0, 255, 255))
                
                # Mask Scores (Top Left)
                info_mask = f"Mask: S={metrics['siglip_mask']:.2f} D={metrics['ds_mask']:.2f}"
                draw_text_with_bg(model_img, info_mask, 10, 60, font_scale=0.5)
                
                # BBox Scores (Inside BBox)
                if bbox:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(model_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                    
                    # Center text in bbox
                    info_bbox = f"BBox\nS={metrics['siglip_bbox']:.2f}\nD={metrics['ds_bbox']:.2f}"
                    y0 = by + bh // 2 - 10
                    for line in info_bbox.split('\n'):
                        draw_text_with_bg(model_img, line, bx + 5, y0, font_scale=0.4, bg_color=(0,0,0))
                        y0 += 15
            else:
                draw_text_with_bg(model_img, f"{m_name} (No Data)", 10, 30)

            canvas[y_off:y_off+CELL_SIZE, x_off:x_off+CELL_SIZE] = model_img

    cv2.imwrite(OUTPUT_IMAGE, canvas)
    print(f"Visualization saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    visualize_metrics()
