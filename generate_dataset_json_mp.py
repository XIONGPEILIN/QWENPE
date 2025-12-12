import os
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 配置路径
BASE_DIR = "pico-banana-400k-subject_driven"
AUDIT_DIR = os.path.join(BASE_DIR, "openimages/dino_mask_audit")
OUTPUT_JSON = "dataset_qwen_pe.json"
REF_GT_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_generated")

# 确保输出目录存在
os.makedirs(REF_GT_DIR, exist_ok=True)

# 强制使用 GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def _load_mask_tensor(path, size=None):
    try:
        m = Image.open(path).convert("L")
        transform = transforms.ToTensor()
        t = transform(m)
        t = t.unsqueeze(0)
        # 移至 GPU
        t = t.to(DEVICE)
        
        if size is not None:
            target_h, target_w = size[1], size[0]
            if t.shape[-2:] != (target_h, target_w):
                t = F.interpolate(t, size=(target_h, target_w), mode='nearest')
        return t
    except:
        return None

def _get_bbox_mask_tensor(mask_tensor):
    if mask_tensor is None: return None
    # 保持在 GPU 上运算
    pts = torch.nonzero(mask_tensor[0, 0] > 0.5, as_tuple=True)
    if len(pts[0]) == 0:
        return torch.zeros_like(mask_tensor)
    
    y_min, y_max = pts[0].min().item(), pts[0].max().item()
    x_min, x_max = pts[1].min().item(), pts[1].max().item()
    
    patch_size = 16
    y_min_exp = (y_min // patch_size) * patch_size
    x_min_exp = (x_min // patch_size) * patch_size
    y_max_exp = ((y_max + patch_size) // patch_size) * patch_size - 1
    x_max_exp = ((x_max + patch_size) // patch_size) * patch_size - 1
    
    H, W = mask_tensor.shape[-2:]
    y_min_exp = max(0, y_min_exp)
    y_max_exp = min(H - 1, y_max_exp)
    x_min_exp = max(0, x_min_exp)
    x_max_exp = min(W - 1, x_max_exp)
    
    box_mask = torch.zeros_like(mask_tensor)
    box_mask[:, :, y_min_exp : y_max_exp + 1, x_min_exp : x_max_exp + 1] = 1.0
    return box_mask

def process_one_file(audit_file):
    REL_ROOT = os.path.join(BASE_DIR, "openimages")
    try:
        with open(audit_file, 'r') as f:
            audit_data = json.load(f)
        
        results = audit_data.get("results", {})
        global_res = results.get("global", {})
        
        # 1. 筛选 background_bbox_sim > 0.9
        bg_sim = global_res.get("background_bbox_sim", 0)
        if bg_sim <= 0.9:
            return None

        # Load Log to get Image Size and Paths
        log_rel_path = audit_data.get("log_path")
        if not log_rel_path: return None
        log_full_path = os.path.join(BASE_DIR, log_rel_path)
        if not os.path.exists(log_full_path): return None

        with open(log_full_path, 'r') as f:
            log_data = json.load(f)

        bg_rel_path = log_data.get("original_item", {}).get("local_input_image")
        edit_rel_path = log_data.get("original_item", {}).get("output_image")
        if not bg_rel_path or not edit_rel_path: return None
        
        edit_full = os.path.join(BASE_DIR, edit_rel_path)
        if not os.path.exists(edit_full): return None
        
        # We need image size for mask loading
        with Image.open(edit_full) as img:
            img_size = img.size # (W, H)

        # 2. Accumulate Masks (Additive Logic)
        precise_accum = None
        bbox_accum = None
        
        for kind in ["add", "remove"]:
            kind_res = results.get(kind, {})
            sub_results = kind_res.get("sub_mask_results", [])
            has_valid_subs = False
            
            # Try sub-masks first
            for sub in sub_results:
                if sub.get("cos_sim", 0) < 0.9: # MODIFIED
                    mask_rel = sub.get("mask_path")
                    if mask_rel:
                        mask_full = os.path.join(BASE_DIR, mask_rel)
                        if os.path.exists(mask_full):
                            t = _load_mask_tensor(mask_full, size=img_size)
                            if t is not None:
                                has_valid_subs = True
                                
                                # Add to Precise Accum
                                if precise_accum is None: precise_accum = t
                                else: precise_accum = torch.max(precise_accum, t)
                                
                                # Add to BBox Accum
                                b_t = _get_bbox_mask_tensor(t)
                                if bbox_accum is None: bbox_accum = b_t
                                else: bbox_accum = torch.max(bbox_accum, b_t)
            
            # Fallback to Merged Mask
            if not has_valid_subs:
                merged_rel = global_res.get(f"{kind}_mask_path") or kind_res.get("kind_merged_mask_path")
                if merged_rel:
                    merged_full = os.path.join(BASE_DIR, merged_rel)
                    if os.path.exists(merged_full):
                        t = _load_mask_tensor(merged_full, size=img_size)
                        if t is not None:
                            if precise_accum is None: precise_accum = t
                            else: precise_accum = torch.max(precise_accum, t)
                            
                            b_t = _get_bbox_mask_tensor(t)
                            if bbox_accum is None: bbox_accum = b_t
                            else: bbox_accum = torch.max(bbox_accum, b_t)

        if precise_accum is None or bbox_accum is None:
            return None

        # 3. Extract BBox from bbox_accum
        pts = torch.nonzero(bbox_accum[0, 0] > 0.5, as_tuple=True)
        if len(pts[0]) == 0: return None
        
        y_min, y_max = pts[0].min().item(), pts[0].max().item()
        x_min, x_max = pts[1].min().item(), pts[1].max().item()
        
        # PIL Crop: (left, upper, right, lower) -> (x_min, y_min, x_max+1, y_max+1)
        crop_box = (x_min, y_min, x_max + 1, y_max + 1)
        
        # 4. Generate Ref GT and Final Mask
        item_idx = audit_data.get("item_idx")
        ref_gt_filename = f"ref_gt_{item_idx}.png"
        ref_gt_full_path = os.path.join(REF_GT_DIR, ref_gt_filename)
        
        # Mask filename
        new_mask_filename = f"mask_combined_{item_idx}.png"
        new_mask_full_path = os.path.join(REF_GT_DIR, new_mask_filename)
        
        # Only process if output doesn't exist
        if not os.path.exists(ref_gt_full_path):
            edit_img = Image.open(edit_full).convert("RGB")
            
            # Crop Image
            gt_crop = edit_img.crop(crop_box)
            
            # Crop Precise Mask
            # precise_accum is [1, 1, H, W] tensor on GPU
            # Slice on GPU
            mask_slice = precise_accum[0, 0, y_min:y_max+1, x_min:x_max+1]
            
            # Move to CPU for PIL
            mask_crop_pil = transforms.ToPILImage()(mask_slice.cpu())
            
            # Apply Mask
            ref_gt_img = Image.new("RGB", gt_crop.size, (0, 0, 0))
            ref_gt_img.paste(gt_crop, (0, 0), mask=mask_crop_pil)
            ref_gt_img.save(ref_gt_full_path)
            
            # Save the Full Precise Mask (for back_mask)
            full_mask_pil = transforms.ToPILImage()(precise_accum[0].cpu())
            full_mask_pil.save(new_mask_full_path)

        # 5. Construct Entry
        original_text = log_data.get("original_item", {}).get("text", "")
        if not original_text:
             original_text = log_data.get("instruction", "")
        
        prompt = f"Picture 1 is the image to modify. {original_text}"
        
        # Path processing
        bg_full = os.path.join(BASE_DIR, bg_rel_path)
        bg_rel_path = os.path.relpath(bg_full, REL_ROOT)
        edit_rel_path = os.path.relpath(edit_full, REL_ROOT)
        ref_gt_rel_path = os.path.relpath(ref_gt_full_path, REL_ROOT)
        final_mask_rel_path = os.path.relpath(new_mask_full_path, REL_ROOT)

        return {
            "prompt": prompt,
            "image": edit_rel_path,        # GT
            "edit_image": [bg_rel_path],   # Source
            "ref_gt": ref_gt_rel_path,
            "back_mask": final_mask_rel_path
        }

    except Exception as e:
        return None

def main():
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_dino_audit.json"))
    print(f"Found {len(audit_files)} audit files. Processing with {cpu_count()} CPUs...")
    
    # 注意：在多进程中使用 CUDA 需要注意 spawn 方法
    # 如果简单 Pool 可能报错。对于简单任务，单进程 GPU 可能比 多进程 GPU 更稳定且不慢（因为 I/O 也是瓶颈）
    # 或者设置 start method
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    dataset = []
    
    # 减少并发数以避免显存溢出，假设每个进程占用少量显存
    # 如果 GPU 显存大，可以多开。这里保守设为 8。
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_one_file, audit_files), total=len(audit_files)))
    
    dataset = [r for r in results if r is not None]

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated dataset with {len(dataset)} entries saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
