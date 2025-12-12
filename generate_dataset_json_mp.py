import os
import json
import glob
import numpy as np
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

def process_one_file(audit_file):
    try:
        with open(audit_file, 'r') as f:
            audit_data = json.load(f)
        
        results = audit_data.get("results", {})
        global_res = results.get("global", {})
        
        # 1. 筛选 background_bbox_sim > 0.9
        bg_sim = global_res.get("background_bbox_sim", 0)
        if bg_sim <= 0.9:
            return None

        # Helper to process a specific kind (add or remove)
        def get_cleaned_mask_array(kind):
            kind_res = results.get(kind)
            if not kind_res:
                return None, None
            
            # Get merged mask path
            merged_rel = global_res.get(f"{kind}_mask_path")
            if not merged_rel:
                merged_rel = kind_res.get("kind_merged_mask_path")
            if not merged_rel:
                return None, None
                
            merged_full = os.path.join(BASE_DIR, merged_rel)
            if not os.path.exists(merged_full):
                return None, None
            
            try:
                img = Image.open(merged_full).convert("L")
                arr = np.array(img)
                # Make writable
                arr = arr.copy()
            except:
                return None, None
                
            # Subtract unchanged sub-masks
            sub_results = kind_res.get("sub_mask_results", [])
            modified = False
            
            for sub in sub_results:
                if sub.get("cos_sim", 0) >= 0.9: # Unchanged -> Remove
                    sub_rel = sub.get("mask_path")
                    if not sub_rel: continue
                    sub_full = os.path.join(BASE_DIR, sub_rel)
                    if os.path.exists(sub_full):
                        try:
                            sub_img = Image.open(sub_full).convert("L")
                            if sub_img.size != img.size:
                                sub_img = sub_img.resize(img.size, Image.NEAREST)
                            sub_arr = np.array(sub_img)
                            
                            # Subtract
                            arr[sub_arr > 128] = 0
                            modified = True
                        except:
                            pass
            
            return arr, modified

        # 2. Get masks for both Add and Remove
        add_arr, add_mod = get_cleaned_mask_array("add")
        rem_arr, rem_mod = get_cleaned_mask_array("remove")
        
        if add_arr is None and rem_arr is None:
            return None
            
        # 3. Combine Masks
        final_arr = None
        
        # Init final_arr with add or rem
        if add_arr is not None:
            final_arr = add_arr
        
        if rem_arr is not None:
            if final_arr is None:
                final_arr = rem_arr
            else:
                # Resize if necessary (shouldn't be, but safety first)
                if rem_arr.shape != final_arr.shape:
                    # Skip complex resizing for now, assume same size as they come from same pipeline
                    pass 
                else:
                    # Union: pixelwise OR (max)
                    # final_arr = np.maximum(final_arr, rem_arr)
                    # Use boolean logic then convert back to uint8 to avoid overflow wrap-around
                    final_arr = ((final_arr > 128) | (rem_arr > 128)).astype(np.uint8) * 255

        # 4. Empty Check
        if final_arr is None or np.max(final_arr) < 10:
            return None

        # 5. Save Final Combined Mask
        # Always save as a new cleaned mask because it might be a combination or cleaned
        item_idx = audit_data.get("item_idx")
        new_mask_filename = f"mask_combined_{item_idx}.png"
        new_mask_full_path = os.path.join(REF_GT_DIR, new_mask_filename)
        
        # Only save if not exists (or overwrite? overwrite is safer for updates)
        # But for mp safety with check exists
        if not os.path.exists(new_mask_full_path):
             Image.fromarray(final_arr).save(new_mask_full_path)
        
        final_mask_rel_path = os.path.relpath(new_mask_full_path, BASE_DIR)

        # 6. Load Log
        log_rel_path = audit_data.get("log_path")
        if not log_rel_path:
            return None
        
        log_full_path = os.path.join(BASE_DIR, log_rel_path)
        if not os.path.exists(log_full_path):
            return None

        with open(log_full_path, 'r') as f:
            log_data = json.load(f)

        bg_rel_path = log_data.get("original_item", {}).get("local_input_image")
        edit_rel_path = log_data.get("original_item", {}).get("output_image")
        
        if not bg_rel_path or not edit_rel_path:
            return None

        # 7. Generate Ref GT
        # Ref GT 应该基于 GT (Result) 图生成，即 log 中的 output_image
        edit_full_path = os.path.join(BASE_DIR, edit_rel_path)
        if not os.path.exists(edit_full_path):
            return None
            
        ref_gt_filename = f"ref_gt_{item_idx}.png"
        ref_gt_full_path = os.path.join(REF_GT_DIR, ref_gt_filename)
        ref_gt_rel_path = os.path.relpath(ref_gt_full_path, BASE_DIR)

        if not os.path.exists(ref_gt_full_path):
            try:
                edit_img = Image.open(edit_full_path).convert("RGB")
                mask_img = Image.fromarray(final_arr)
                
                if mask_img.size != edit_img.size:
                    mask_img = mask_img.resize(edit_img.size, Image.NEAREST)

                ref_gt_img = Image.new("RGB", edit_img.size, (0, 0, 0))
                ref_gt_img.paste(edit_img, (0, 0), mask=mask_img)
                ref_gt_img.save(ref_gt_full_path)
            except Exception:
                return None

        # 8. 构建 Entry
        # 优先使用 original_item.text，如果没有则回退到 instruction
        original_text = log_data.get("original_item", {}).get("text", "")
        if not original_text:
             original_text = log_data.get("instruction", "")
        
        prompt = f"Picture 1 is the image to modify. {original_text}"

        # 修正映射：image 是 GT (Output), edit_image 是 Source (Input)
        return {
            "prompt": prompt,
            "image": edit_rel_path,        # GT (Output Image)
            "edit_image": [bg_rel_path],   # Source (Input Image)
            "ref_gt": ref_gt_rel_path,     # GT Cutout
            "back_mask": final_mask_rel_path
        }

    except Exception:
        return None

def main():
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_dino_audit.json"))
    print(f"Found {len(audit_files)} audit files. Processing with {cpu_count()} CPUs...")

    dataset = []
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_one_file, audit_files), total=len(audit_files)))
    
    dataset = [r for r in results if r is not None]

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated dataset with {len(dataset)} entries saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
