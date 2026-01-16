import pandas as pd
import json
import os
import argparse
import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

def visualize(csv_path, json_path, pred_dir, gt_base_dir, output_prefix, sort_by):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    # Create mappings
    file_to_entry = {}
    for entry in dataset:
        if entry.get('edit_image'):
            fname = os.path.basename(entry['edit_image'][0])
            file_to_entry[fname] = entry

    # 2. Sort results
    if sort_by in df.columns:
        ascending = True if 'dreamsim' in sort_by or 'l1' in sort_by or 'l2' in sort_by else False
        df_sorted = df.sort_values(by=sort_by, ascending=ascending)
    else:
        df_sorted = df.sort_values(by='dreamsim')
        
    # Selection logic for range
    start_idx = args.start_idx
    num_items = args.num_items
    selected_data = df_sorted.iloc[start_idx : start_idx + num_items]

    def extract_bbox(mask_np):
        rows = np.any(mask_np > 128, axis=1)
        cols = np.any(mask_np > 128, axis=0)
        if not np.any(rows) or not np.any(cols): return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (cmin, rmin, cmax + 1, rmax + 1)

    def create_grid(data, title, save_path):
        num_to_plot = len(data)
        if num_to_plot == 0: 
            print("No data to plot.")
            return
        fig, axes = plt.subplots(num_to_plot, 4, figsize=(24, 6 * num_to_plot))
        # Handle case with only 1 item (axes becomes 1D)
        if num_to_plot == 1: axes = np.expand_dims(axes, axis=0)
        
        fig.suptitle(title, fontsize=40, y=1.02)
        
        data_list = data.to_dict('records')
        for i in range(num_to_plot):
            item = data_list[i]
            fname = item['filename']
            entry = file_to_entry.get(fname, {})
            
            # Paths
            pred_path = os.path.join(pred_dir, fname)
            gt_path = os.path.join(gt_base_dir, entry.get('image', ""))
            mask_rel = entry.get('back_mask', "")
            
            try:
                img_p = Image.open(pred_path).convert("RGB")
                img_g = Image.open(gt_path).convert("RGB")
                if img_p.size != img_g.size:
                    img_p = img_p.resize(img_g.size, Image.LANCZOS)
                
                mask_pil = None
                if mask_rel:
                    m_path = os.path.join(gt_base_dir, mask_rel)
                    if os.path.exists(m_path):
                        mask_pil = Image.open(m_path).convert("L").resize(img_p.size, Image.NEAREST)

                # 1. Column 1: Pred with Mask Overlay
                img_overlay = img_p.copy()
                if mask_pil:
                    red = Image.new("RGBA", img_p.size, (255, 0, 0, 64)) # Increased transparency (lower alpha)
                    mask_alpha = (np.array(mask_pil) > 128).astype(np.uint8) * 64
                    red.putalpha(Image.fromarray(mask_alpha))
                    img_overlay = Image.alpha_composite(img_p.convert("RGBA"), red).convert("RGB")
                
                axes[i, 0].imshow(img_overlay)
                title_str = f"{fname}\nGlobal DS: {item.get('dreamsim', 0):.4f}"
                axes[i, 0].set_title(title_str, fontsize=16)
                axes[i, 0].axis('off')

                # 2. Column 2: GT
                axes[i, 1].imshow(img_g)
                axes[i, 1].set_title("Ground Truth", fontsize=16)
                axes[i, 1].axis('off')

                # 3. Column 3: BBox Crop
                if mask_pil:
                    bbox = extract_bbox(np.array(mask_pil))
                    if bbox:
                        crop_p = img_p.crop(bbox)
                        axes[i, 2].imshow(crop_p)
                        axes[i, 2].set_title(f"BBox Crop\nDS_BBox: {item.get('dreamsim_bbox', 0):.4f}", fontsize=16)
                    else:
                        axes[i, 2].text(0.5, 0.5, "No BBox", ha='center')
                axes[i, 2].axis('off')

                # 4. Column 4: Masked Pred (Black BG)
                if mask_pil:
                    black = Image.new("RGB", img_p.size, (0, 0, 0))
                    masked_p = Image.composite(img_p, black, mask_pil)
                    axes[i, 3].imshow(masked_p)
                    m_info = f"CLIP_M: {item.get('clip_i_mask', 0):.3f} | DINO_M: {item.get('dino_mask', 0):.3f}\nDS_Mask: {item.get('dreamsim_mask', 0):.4f}"
                    axes[i, 3].set_title(f"Masked (Foreground)\n{m_info}", fontsize=16)
                axes[i, 3].axis('off')

            except Exception as e:
                print(f"Error drawing {fname}: {e}")
                for j in range(4): axes[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()

    # 3. Generate requested range
    range_title = f"Results {start_idx} to {start_idx + num_items} (Sorted by {sort_by})"
    range_path = f"{output_prefix}_range_{start_idx}_{start_idx + num_items}.png"
    create_grid(selected_data, range_title, range_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--out", default="viz_results")
    parser.add_argument("--sort_by", default="dreamsim_mask", help="Metric to sort by")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in sorted results")
    parser.add_argument("--num_items", type=int, default=10, help="Number of items to visualize")
    args = parser.parse_args()

    visualize(args.csv, args.json, args.pred_dir, args.gt_dir, args.out, args.sort_by)