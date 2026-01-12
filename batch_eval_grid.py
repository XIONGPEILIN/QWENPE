#!/usr/bin/env python
import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from queue import Empty
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, pipeline
from dreamsim import dreamsim as load_dreamsim_lib

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CKPT_NAME = "2011-ste-28000"
BASE_GRID_DIR = Path("compare/grid_search")
JSON_PATH = "dataset_qwen_pe_top1000.json"
GT_BASE_DIR = "/home/yanai-lab/xiong-p/ssd/xiong-p/qwenpe/pico-banana-400k-subject_driven/openimages"
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 7]

# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------
class EvalDataset(Dataset):
    def __init__(self, json_path, pred_dir, gt_base_dir):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.pred_dir = pred_dir
        self.gt_base_dir = gt_base_dir
        self.valid_indices = self._filter_valid()

    def _filter_valid(self):
        valid = []
        for i, entry in enumerate(self.data):
            if not entry.get('edit_image'): continue
            fname = os.path.basename(entry['edit_image'][0])
            p_path = os.path.join(self.pred_dir, fname)
            g_path = os.path.join(self.gt_base_dir, entry['image'])
            
            if os.path.exists(p_path) and os.path.exists(g_path):
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        entry = self.data[real_idx]
        
        fname = os.path.basename(entry['edit_image'][0])
        p_path = os.path.join(self.pred_dir, fname)
        g_path = os.path.join(self.gt_base_dir, entry['image'])
        m_path = os.path.join(self.gt_base_dir, entry['back_mask']) if entry.get('back_mask') else None
        
        img_p = Image.open(p_path).convert("RGB")
        img_g = Image.open(g_path).convert("RGB")
        
        if img_p.size != img_g.size:
            img_p = img_p.resize(img_g.size, Image.LANCZOS)
            
        mask_p = None
        if m_path and os.path.exists(m_path):
            mask_p = Image.open(m_path).convert("L").resize(img_p.size, Image.NEAREST)
            
        return {
            "filename": fname,
            "img_p": img_p,
            "img_g": img_g,
            "mask_p": mask_p,
            "prompt": entry.get('prompt', "")
        }

# -----------------------------------------------------------------------------
# Persistent Worker
# -----------------------------------------------------------------------------
def persistent_gpu_worker(gpu_id, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    print(f"[GPU {gpu_id}] Initializing Models (SigLIP, DINO, DreamSim)...")
    try:
        # SigLIP2
        siglip_ckpt = "google/siglip2-large-patch16-512"
        siglip_model = AutoModel.from_pretrained(siglip_ckpt, torch_dtype=torch.bfloat16).to(device).eval()
        siglip_processor = AutoProcessor.from_pretrained(siglip_ckpt)

        # DINOv3
        dinov3_pipe = pipeline(
            model="facebook/dinov3-vit7b16-pretrain-lvd1689m",
            task="image-feature-extraction",
            device=0, 
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

        # DreamSim
        ds_res = load_dreamsim_lib(pretrained=True)
        ds_model = ds_res[0] if isinstance(ds_res, tuple) else ds_res
        ds_preprocess = ds_res[1] if isinstance(ds_res, tuple) else ds_model.preprocess
        ds_model = ds_model.to(device).eval()
    except Exception as e:
        print(f"[GPU {gpu_id}] FATAL: Model Init Failed: {e}")
        return

    def extract_bbox(mask_pil):
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 128, axis=1); cols = np.any(mask_np > 128, axis=0)
        if not np.any(rows) or not np.any(cols): return None
        rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
        return (cmin, rmin, cmax + 1, rmax + 1)

    print(f"[GPU {gpu_id}] Ready for tasks.")

    while True:
        try:
            task = task_queue.get(timeout=2)
        except Empty:
            continue
        
        if task is None:
            print(f"[GPU {gpu_id}] Shutting down.")
            break

        task_id, flat_dir, output_csv = task
        print(f"[GPU {gpu_id}] Processing: {task_id}")

        try:
            dataset = EvalDataset(JSON_PATH, str(flat_dir), GT_BASE_DIR)
            results = []

            for i in range(len(dataset)):
                item = dataset[i]
                filename, img_p, img_g, mask_p, prompt = item["filename"], item["img_p"], item["img_g"], item["mask_p"], item["prompt"]
                row = {"filename": filename, "prompt": prompt}

                # --- Metrics Logic ---
                # SigLIP
                inputs_img = siglip_processor(images=img_p, return_tensors="pt").to(device)
                inputs_gt = siglip_processor(images=img_g, return_tensors="pt").to(device)
                if siglip_model.dtype == torch.bfloat16:
                    inputs_img["pixel_values"] = inputs_img["pixel_values"].to(torch.bfloat16)
                    inputs_gt["pixel_values"] = inputs_gt["pixel_values"].to(torch.bfloat16)
                
                with torch.inference_mode():
                    emb_img = siglip_model.get_image_features(**inputs_img)
                    emb_gt = siglip_model.get_image_features(**inputs_gt)
                    emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
                    emb_gt = emb_gt / emb_gt.norm(dim=-1, keepdim=True)
                    row["clip_i"] = torch.sum(emb_img.float() * emb_gt.float(), dim=-1).item()
                    
                    if prompt:
                        inputs_text = siglip_processor(text=prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                        emb_text = siglip_model.get_text_features(**inputs_text)
                        emb_text = emb_text / emb_text.norm(dim=-1, keepdim=True)
                        row["clip_t"] = torch.sum(emb_img.float() * emb_text.float(), dim=-1).item()

                # DINO
                with torch.inference_mode():
                    feat_img = dinov3_pipe(img_p)
                    feat_gt = dinov3_pipe(img_g)
                    t_img = torch.tensor(feat_img[0]).mean(dim=0 if torch.tensor(feat_img[0]).ndim==2 else 1, keepdim=True)
                    t_gt = torch.tensor(feat_gt[0]).mean(dim=0 if torch.tensor(feat_gt[0]).ndim==2 else 1, keepdim=True)
                    row["dino"] = F.cosine_similarity(t_img.float(), t_gt.float(), dim=-1).item()

                # DreamSim
                with torch.inference_mode():
                    t_ds_p = ds_preprocess(img_p).to(device)
                    t_ds_g = ds_preprocess(img_g).to(device)
                    row["dreamsim"] = ds_model(t_ds_p, t_ds_g).item()

                # Mask/BBox Logic
                if mask_p:
                    bbox = extract_bbox(mask_p)
                    if bbox:
                        img_p_crop = img_p.crop(bbox)
                        img_g_crop = img_g.crop(bbox)
                        
                        # SigLIP BBox
                        inp_b_p = siglip_processor(images=img_p_crop, return_tensors="pt").to(device)
                        inp_b_g = siglip_processor(images=img_g_crop, return_tensors="pt").to(device)
                        if siglip_model.dtype == torch.bfloat16:
                            inp_b_p["pixel_values"] = inp_b_p["pixel_values"].to(torch.bfloat16)
                            inp_b_g["pixel_values"] = inp_b_g["pixel_values"].to(torch.bfloat16)
                        with torch.inference_mode():
                            e_b_p = siglip_model.get_image_features(**inp_b_p); e_b_p /= e_b_p.norm(dim=-1, keepdim=True)
                            e_b_g = siglip_model.get_image_features(**inp_b_g); e_b_g /= e_b_g.norm(dim=-1, keepdim=True)
                            row["clip_i_bbox"] = torch.sum(e_b_p.float() * e_b_g.float(), dim=-1).item()
                        
                        # DINO BBox
                        with torch.inference_mode():
                            f_b_p = dinov3_pipe(img_p_crop)
                            f_b_g = dinov3_pipe(img_g_crop)
                            t_b_p = torch.tensor(f_b_p[0]).mean(dim=0 if torch.tensor(f_b_p[0]).ndim==2 else 1, keepdim=True)
                            t_b_g = torch.tensor(f_b_g[0]).mean(dim=0 if torch.tensor(f_b_g[0]).ndim==2 else 1, keepdim=True)
                            row["dino_bbox"] = F.cosine_similarity(t_b_p.float(), t_b_g.float(), dim=-1).item()

                        # DS BBox
                        with torch.inference_mode():
                            t_d_b_p = ds_preprocess(img_p_crop).to(device)
                            t_d_b_g = ds_preprocess(img_g_crop).to(device)
                            row["dreamsim_bbox"] = ds_model(t_d_b_p, t_d_b_g).item()

                    # Mask
                    black = Image.new("RGB", img_p.size, (0, 0, 0))
                    img_p_m = Image.composite(img_p, black, mask_p)
                    img_g_m = Image.composite(img_g, black, mask_p)
                    
                    # SigLIP Mask
                    inp_m_p = siglip_processor(images=img_p_m, return_tensors="pt").to(device)
                    inp_m_g = siglip_processor(images=img_g_m, return_tensors="pt").to(device)
                    if siglip_model.dtype == torch.bfloat16:
                        inp_m_p["pixel_values"] = inp_m_p["pixel_values"].to(torch.bfloat16)
                        inp_m_g["pixel_values"] = inp_m_g["pixel_values"].to(torch.bfloat16)
                    with torch.inference_mode():
                        e_m_p = siglip_model.get_image_features(**inp_m_p); e_m_p /= e_m_p.norm(dim=-1, keepdim=True)
                        e_m_g = siglip_model.get_image_features(**inp_m_g); e_m_g /= e_m_g.norm(dim=-1, keepdim=True)
                        row["clip_i_mask"] = torch.sum(e_m_p.float() * e_m_g.float(), dim=-1).item()

                    # DINO Mask
                    with torch.inference_mode():
                        f_m_p = dinov3_pipe(img_p_m)
                        f_m_g = dinov3_pipe(img_g_m)
                        t_m_p = torch.tensor(f_m_p[0]).mean(dim=0 if torch.tensor(f_m_p[0]).ndim==2 else 1, keepdim=True)
                        t_m_g = torch.tensor(f_m_g[0]).mean(dim=0 if torch.tensor(f_m_g[0]).ndim==2 else 1, keepdim=True)
                        row["dino_mask"] = F.cosine_similarity(t_m_p.float(), t_m_g.float(), dim=-1).item()

                    # DS Mask
                    with torch.inference_mode():
                        t_d_m_p = ds_preprocess(img_p_m).to(device)
                        t_d_m_g = ds_preprocess(img_g_m).to(device)
                        row["dreamsim_mask"] = ds_model(t_d_m_p, t_d_m_g).item()


                # L1/L2 Calculation (Full Res)
                with torch.inference_mode():
                    to_tensor = transforms.ToTensor()
                    t_p = to_tensor(img_p).to(device)
                    t_g = to_tensor(img_g).to(device)
                    
                    diff_sq = (t_p - t_g) ** 2
                    diff_abs = torch.abs(t_p - t_g)
                    
                    row["l2"] = torch.mean(diff_sq).item()
                    row["l1"] = torch.mean(diff_abs).item()

                    if mask_p:
                        t_m = to_tensor(mask_p).to(device)
                        m_sum = t_m.sum()
                        if m_sum > 0:
                            row["l2_in_mask"] = (diff_sq * t_m).sum().item() / (m_sum.item() * 3)
                            row["l1_in_mask"] = (diff_abs * t_m).sum().item() / (m_sum.item() * 3)
                        
                        t_m_inv = 1.0 - t_m
                        m_inv_sum = t_m_inv.sum()
                        if m_inv_sum > 0:
                            row["l2_out_mask"] = (diff_sq * t_m_inv).sum().item() / (m_inv_sum.item() * 3)
                            row["l1_out_mask"] = (diff_abs * t_m_inv).sum().item() / (m_inv_sum.item() * 3)

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

                results.append(row)
            
            # Save Results
            if results:
                df = pd.DataFrame(results)
                df.to_csv(output_csv, index=False)
                
                m_map = {
                    "clip_i": "SigLIP2_I", "clip_t": "SigLIP2_T", "dino": "DINO", "dreamsim": "DS",
                    "clip_i_bbox": "SigLIP2_I_BBox", "dino_bbox": "DINO_BBox", "dreamsim_bbox": "DS_BBox",
                    "clip_i_mask": "SigLIP2_I_Mask", "dino_mask": "DINO_Mask", "dreamsim_mask": "DS_Mask",
                    "l2": "MSE", "l1": "MAE",
                    "l2_in_mask": "MSE_InMask", "l1_in_mask": "MAE_InMask",
                    "l2_out_mask": "MSE_OutMask", "l1_out_mask": "MAE_OutMask",
                    "l2_in_bbox": "MSE_InBBox", "l1_in_bbox": "MAE_InBBox",
                    "l2_out_bbox": "MSE_OutBBox", "l1_out_bbox": "MAE_OutBBox"
                }
                summary = {m_map[k]: float(df[k].mean()) for k, v in m_map.items() if k in df.columns}
                summary_path = str(output_csv).replace(".csv", "_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=4)
                
                print(f"[GPU {gpu_id}] Finished {task_id}")
            else:
                print(f"[GPU {gpu_id}] No valid data for {task_id}")

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR processing {task_id}: {e}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def prepare_flat_directory(task_name, src_dir):
    flat_dir = Path(f"temp_flat_{task_name}")
    if flat_dir.exists():
        shutil.rmtree(flat_dir)
    flat_dir.mkdir(parents=True)
    
    with open(JSON_PATH, 'r') as f:
        dataset_data = json.load(f)
    
    count = 0
    for sample_folder in src_dir.iterdir():
        if not sample_folder.is_dir(): continue
        try:
            idx = int(sample_folder.name)
            if idx < 0 or idx >= len(dataset_data): continue
            
            orig_rel_path = dataset_data[idx]['edit_image'][0]
            filename = os.path.basename(orig_rel_path)
            output_img = sample_folder / "output.png"
            if output_img.exists():
                shutil.copy(output_img, flat_dir / filename)
                count += 1
        except ValueError:
            continue
    return flat_dir if count > 0 else None

def main():
    if not BASE_GRID_DIR.exists():
        print(f"Base dir {BASE_GRID_DIR} does not exist.")
        return

    tasks = []
    print(f"Scanning {BASE_GRID_DIR}...")
    for child in BASE_GRID_DIR.iterdir():
        if child.is_dir() and child.name.startswith(CKPT_NAME):
            task_name = child.name
            flat_dir = prepare_flat_directory(task_name, child)
            if flat_dir:
                output_csv = child / "evaluation_results.csv"
                tasks.append((task_name, flat_dir, output_csv))
    
    print(f"Total tasks: {len(tasks)}")
    if not tasks: return

    num_workers = min(len(tasks), len(AVAILABLE_GPUS))
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for t in tasks: task_queue.put(t)
    for _ in range(num_workers): task_queue.put(None)

    print(f"Starting {num_workers} workers...")
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=persistent_gpu_worker, args=(AVAILABLE_GPUS[i], task_queue, result_queue))
        p.start(); workers.append(p)

    for p in workers: p.join()
    print("All tasks completed.")

    # Cleanup temp dirs
    print("Cleaning up temp directories...")
    for t in tasks:
        flat_dir = t[1]
        if flat_dir.exists(): shutil.rmtree(flat_dir)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
