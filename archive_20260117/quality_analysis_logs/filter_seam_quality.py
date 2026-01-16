import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing

# --- Configuration ---
JSON_PATH = "dataset_qwen_pe_test.json"
BASE_DIR = "pico-banana-400k-subject_driven/openimages"
OUTPUT_REPORT = "seam_quality_report.json"
OUTPUT_VISUALIZATION = "seam_quality_inspection.jpg"

# Visualization: Top 20 Worst vs Top 20 Best
VISUALIZE_COUNT = 20

def calculate_seam_score(args):
    """
    Calculates the 'Seam Gradient Difference Score'.
    Score = Mean(|Grad_Edit - Grad_Target|) in the Seam Region.
    Cancels out background textures, highlighting only artificial cuts.
    """
    idx, item = args
    try:
        edit_rel = item["edit_image"][0]
        mask_rel = item["back_mask"]
        target_rel = item["image"]
        
        edit_path = os.path.join(BASE_DIR, edit_rel)
        mask_path = os.path.join(BASE_DIR, mask_rel)
        target_path = os.path.join(BASE_DIR, target_rel)
        
        if not (os.path.exists(edit_path) and os.path.exists(mask_path) and os.path.exists(target_path)):
            return None
            
        # Load Images
        img_edit = cv2.imread(edit_path)
        img_target = cv2.imread(target_path)
        img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img_edit is None or img_target is None or img_mask is None:
            return None
            
        # Ensure sizes match (Target is truth)
        h, w = img_target.shape[:2]
        if img_edit.shape[:2] != (h, w):
            img_edit = cv2.resize(img_edit, (w, h))
        if img_mask.shape != (h, w):
            img_mask = cv2.resize(img_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 1. Define Wide Seam Region (9x9, 8 iters)
        kernel = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(img_mask, kernel, iterations=8)
        eroded = cv2.erode(img_mask, kernel, iterations=8)
        seam_mask = cv2.subtract(dilated, eroded)
        
        # Normalize Seam Mask (0.0 - 1.0)
        seam_map = seam_mask.astype(np.float32) / 255.0
        seam_pixel_count = np.sum(seam_map)
        
        # If mask is empty or too small, skip
        if seam_pixel_count < 10: 
            return (idx, 0.0, item, edit_path, mask_path, target_path)

        # 2. Calculate Gradient Magnitude for BOTH images
        def get_gradient_magnitude(img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)

        grad_edit = get_gradient_magnitude(img_edit)
        grad_target = get_gradient_magnitude(img_target)
        
        # 3. Calculate Gradient Difference
        # This cancels out texture present in both (Background)
        # and highlights edges present in only one (Ghosting/Cuts)
        grad_diff = np.abs(grad_edit - grad_target)
        
        # 4. Score: Average Gradient Diff on Seam
        masked_diff = grad_diff * seam_map
        score = np.sum(masked_diff) / seam_pixel_count
        
        return (idx, float(score), item, edit_path, mask_path, target_path)

    except Exception:
        return None

def main():
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    print(f"Analyzing Gradient Difference for {len(data)} samples...")
    tasks = [(i, item) for i, item in enumerate(data)]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(calculate_seam_score, tasks), total=len(tasks)))
        
    valid_results = [r for r in results if r is not None]
    
    # Sort by Score Descending (Highest Diff = Worst)
    valid_results.sort(key=lambda x: x[1], reverse=True)
    
    # Save Report
    report = [{"rank": i, "index": r[0], "seam_score": r[1], "image": r[2]["image"]} for i, r in enumerate(valid_results)]
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {OUTPUT_REPORT}")
    
    # Visualize
    print("Generating visualization grid...")
    
    worst_samples = valid_results[:VISUALIZE_COUNT]
    best_samples = valid_results[-VISUALIZE_COUNT:]
    samples_to_show = worst_samples + best_samples
    
    cell_size = 200
    cols = 5 # Edit, Target, Mask, DiffHeatmap, Label
    rows = len(samples_to_show)
    
    canvas = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    
    for row_idx, (idx, score, item, path_edit, path_mask, path_target) in enumerate(samples_to_show):
        img_edit = cv2.imread(path_edit)
        img_target = cv2.imread(path_target)
        img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        img_edit = cv2.resize(img_edit, (cell_size, cell_size))
        img_target = cv2.resize(img_target, (cell_size, cell_size))
        img_mask = cv2.resize(img_mask, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
        
        # Recompute Gradient Diff for Viz
        def get_gradient_magnitude(img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)

        grad_edit = get_gradient_magnitude(img_edit)
        grad_target = get_gradient_magnitude(img_target)
        grad_diff = np.abs(grad_edit - grad_target)

        # Seam Mask
        kernel_viz = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(img_mask, kernel_viz, iterations=8)
        eroded = cv2.erode(img_mask, kernel_viz, iterations=8)
        seam_mask = cv2.subtract(dilated, eroded).astype(float) / 255.0
        
        # Resize Diff to match
        # (Already matched via resize above, but double check not needed)
        
        # Viz
        diff_viz = grad_diff * seam_mask
        diff_viz = np.clip(diff_viz * 2, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_viz, cv2.COLORMAP_JET)
        
        mask_bgr = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        
        # Fill
        y = row_idx * cell_size
        canvas[y:y+cell_size, 0:cell_size] = img_edit
        canvas[y:y+cell_size, cell_size:2*cell_size] = img_target
        canvas[y:y+cell_size, 2*cell_size:3*cell_size] = mask_bgr
        canvas[y:y+cell_size, 3*cell_size:4*cell_size] = heatmap
        
        # Info
        info_area = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        label = "WORST" if row_idx < VISUALIZE_COUNT else "BEST"
        color = (0, 0, 255) if row_idx < VISUALIZE_COUNT else (0, 255, 0)
        
        cv2.putText(info_area, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(info_area, f"Score: {score:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        canvas[y:y+cell_size, 4*cell_size:5*cell_size] = info_area

    cv2.imwrite(OUTPUT_VISUALIZATION, canvas)
    print(f"Visualization saved to {OUTPUT_VISUALIZATION}")
    print("Check the heatmap. It now shows where Edit has edges that Target doesn't (or vice versa).")

if __name__ == "__main__":
    main()