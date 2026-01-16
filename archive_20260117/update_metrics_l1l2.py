#!/usr/bin/env python
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torch.multiprocessing as mp
from queue import Empty
from torchvision import transforms

"""
Update Metrics Script.
Calculates only the missing L1/L2 metrics (Full Resolution, In/Out Mask, In/Out BBox)
and appends them to an existing evaluation CSV file.
"""

def extract_bbox(mask_pil):
    mask_np = np.array(mask_pil)
    rows = np.any(mask_np > 128, axis=1); cols = np.any(mask_np > 128, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
    return (cmin, rmin, cmax + 1, rmax + 1)

def worker(gpu_id, task_indices, result_queue, dataset_map, pred_dir, gt_base_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    to_tensor = transforms.ToTensor()
    
    # Pre-define results dict
    results = {}

    for idx, (filename, entry) in enumerate(task_indices):
        try:
            fname = os.path.basename(entry['edit_image'][0])
            p_path = os.path.join(pred_dir, fname)
            g_path = os.path.join(gt_base_dir, entry['image'])
            m_path = os.path.join(gt_base_dir, entry['back_mask']) if entry.get('back_mask') else None

            if not os.path.exists(p_path) or not os.path.exists(g_path):
                continue
            
            # Load Images
            img_p = Image.open(p_path).convert("RGB")
            img_g = Image.open(g_path).convert("RGB")
            
            # Align sizes
            if img_p.size != img_g.size:
                img_p = img_p.resize(img_g.size, Image.LANCZOS)
                
            mask_p = None
            if m_path and os.path.exists(m_path):
                mask_p = Image.open(m_path).convert("L").resize(img_p.size, Image.NEAREST)

            # Calculation
            row = {"filename": filename}
            
            t_p = to_tensor(img_p).to(device)
            t_g = to_tensor(img_g).to(device)
            
            diff_sq = (t_p - t_g) ** 2
            diff_abs = torch.abs(t_p - t_g)
            
            row["l2"] = torch.mean(diff_sq).item()
            row["l1"] = torch.mean(diff_abs).item()

            if mask_p:
                t_m = to_tensor(mask_p).to(device)
                
                # Inside Mask
                m_sum = t_m.sum()
                if m_sum > 0:
                    row["l2_in_mask"] = (diff_sq * t_m).sum().item() / (m_sum.item() * 3)
                    row["l1_in_mask"] = (diff_abs * t_m).sum().item() / (m_sum.item() * 3)
                
                # Outside Mask
                t_m_inv = 1.0 - t_m
                m_inv_sum = t_m_inv.sum()
                if m_inv_sum > 0:
                    row["l2_out_mask"] = (diff_sq * t_m_inv).sum().item() / (m_inv_sum.item() * 3)
                    row["l1_out_mask"] = (diff_abs * t_m_inv).sum().item() / (m_inv_sum.item() * 3)

                # Inside/Outside BBox
                bbox = extract_bbox(mask_p)
                if bbox:
                    cmin, rmin, cmax, rmax = bbox
                    t_b = torch.zeros_like(t_m)
                    t_b[:, rmin:rmax, cmin:cmax] = 1.0
                    
                    b_sum = t_b.sum()
                    if b_sum > 0:
                        row["l2_in_bbox"] = (diff_sq * t_b).sum().item() / (b_sum.item() * 3)
                        row["l1_in_bbox"] = (diff_abs * t_b).sum().item() / (b_sum.item() * 3)
                    
                    t_b_inv = 1.0 - t_b
                    b_inv_sum = t_b_inv.sum()
                    if b_inv_sum > 0:
                        row["l2_out_bbox"] = (diff_sq * t_b_inv).sum().item() / (b_inv_sum.item() * 3)
                        row["l1_out_bbox"] = (diff_abs * t_b_inv).sum().item() / (b_inv_sum.item() * 3)
            
            result_queue.put(row)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to existing evaluation_results.csv")
    parser.add_argument("--json_path", required=True, help="Path to original dataset json")
    parser.add_argument("--pred_dir", required=True, help="Path to prediction directory")
    parser.add_argument("--gt_base_dir", required=True, help="Path to ground truth base directory")
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    print(f"Loading existing CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    print(f"Loading dataset JSON: {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    # Map filename to dataset entry for quick lookup
    # Assuming CSV 'filename' matches os.path.basename(entry['edit_image'][0])
    dataset_map = {}
    for entry in data:
        if entry.get('edit_image'):
            fname = os.path.basename(entry['edit_image'][0])
            dataset_map[fname] = entry

    # Filter tasks: Only process files present in the CSV
    tasks = []
    for fname in df['filename'].unique():
        if fname in dataset_map:
            tasks.append((fname, dataset_map[fname]))
    
    print(f"Total files to process: {len(tasks)}")

    # MP Setup
    gpu_list = [int(x.strip()) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_list)
    indices = np.array_split(tasks, num_gpus)

    ctx = mp.get_context('spawn')
    r_q = ctx.Queue()
    processes = []
    
    print("Starting workers...")
    for i, gid in enumerate(gpu_list):
        # Convert numpy array of objects back to list for pickling
        task_subset = indices[i].tolist() if len(indices[i]) > 0 else []
        p = ctx.Process(target=worker, args=(gid, task_subset, r_q, None, args.pred_dir, args.gt_base_dir))
        p.start()
        processes.append(p)

    new_results = []
    with tqdm(total=len(tasks), desc="Calculating L1/L2") as pbar:
        while len(new_results) < len(tasks):
            try:
                res = r_q.get(timeout=5)
                new_results.append(res)
                pbar.update(1)
            except Empty:
                if not any(p.is_alive() for p in processes): break

    [p.join() for p in processes]

    # Merge Results
    print("Merging results...")
    new_df = pd.DataFrame(new_results)
    
    # Merge new columns into original dataframe
    # Drop old L1/L2 columns if they exist to overwrite them
    cols_to_drop = [c for c in new_df.columns if c in df.columns and c != 'filename']
    if cols_to_drop:
        print(f"Overwriting columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        
    final_df = pd.merge(df, new_df, on="filename", how="left")
    
    # Save CSV
    print(f"Saving updated CSV to: {args.csv_path}")
    final_df.to_csv(args.csv_path, index=False)

    # Save Summary
    m_map = {"clip_i": "SigLIP2_I", "clip_t": "SigLIP2_T", "dino": "DINO", "dreamsim": "DS",
             "clip_i_bbox": "SigLIP2_I_BBox", "dino_bbox": "DINO_BBox", "dreamsim_bbox": "DS_BBox",
             "clip_i_mask": "SigLIP2_I_Mask", "dino_mask": "DINO_Mask", "dreamsim_mask": "DS_Mask",
             "l2": "MSE", "l1": "MAE",
             "l2_in_mask": "MSE_InMask", "l1_in_mask": "MAE_InMask",
             "l2_out_mask": "MSE_OutMask", "l1_out_mask": "MAE_OutMask",
             "l2_in_bbox": "MSE_InBBox", "l1_in_bbox": "MAE_InBBox",
             "l2_out_bbox": "MSE_OutBBox", "l1_out_bbox": "MAE_OutBBox"}
    
    print("\n" + "="*40 + "\nUPDATED SUMMARY\n" + "="*40)
    for k, v in m_map.items():
        if k in final_df.columns: print(f"{v}: {final_df[k].mean():.4f}")
    
    json_path = os.path.splitext(args.csv_path)[0] + "_summary.json"
    with open(json_path, "w") as f:
        json.dump({m_map[k]: float(final_df[k].mean()) for k, v in m_map.items() if k in final_df.columns}, f, indent=4)
    print(f"Updated summary JSON: {json_path}")

if __name__ == "__main__":
    main()
