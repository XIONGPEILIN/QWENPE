import os
import json
import glob
from PIL import Image
from tqdm import tqdm

# 配置路径
BASE_DIR = "pico-banana-400k-subject_driven"
AUDIT_DIR = os.path.join(BASE_DIR, "openimages/dino_mask_audit")
OUTPUT_JSON = "dataset_qwen_pe.json"
REF_GT_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_generated")

os.makedirs(REF_GT_DIR, exist_ok=True)

def process():
    dataset = []
    # 查找所有 dino audit json
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_dino_audit.json"))
    
    print(f"Found {len(audit_files)} audit files.")

    for audit_file in tqdm(audit_files):
        try:
            with open(audit_file, 'r') as f:
                audit_data = json.load(f)
            
            # 1. 筛选 background_bbox_sim > 0.9
            # 注意：有些数据可能没有这个字段，设默认值 0
            sim = audit_data.get("results", {}).get("global", {}).get("background_bbox_sim", 0)
            if sim <= 0.9:
                continue
            
            # 2. 获取 Mask 路径 (必须是 Add 任务)
            # 优先查看 global.add_mask_path
            mask_rel_path = audit_data.get("results", {}).get("global", {}).get("add_mask_path")
            
            # 如果没有，尝试从 add 结构中找
            if not mask_rel_path:
                add_data = audit_data.get("results", {}).get("add")
                if isinstance(add_data, dict):
                    mask_rel_path = add_data.get("kind_merged_mask_path")
            
            if not mask_rel_path:
                # 这是一个 remove 任务或者没有 add mask，跳过
                continue

            # 3. 加载原始 Log
            log_rel_path = audit_data.get("log_path")
            if not log_rel_path:
                continue
            
            log_full_path = os.path.join(BASE_DIR, log_rel_path)
            if not os.path.exists(log_full_path):
                print(f"Log file not found: {log_full_path}")
                continue

            with open(log_full_path, 'r') as f:
                log_data = json.load(f)

            # 4. 提取图像路径
            # 背景图
            bg_rel_path = log_data.get("original_item", {}).get("local_input_image")
            # 编辑后的图 (Result)
            edit_rel_path = log_data.get("original_item", {}).get("output_image")
            
            if not bg_rel_path or not edit_rel_path:
                continue

            # 5. 生成 ref_gt
            mask_full_path = os.path.join(BASE_DIR, mask_rel_path)
            edit_full_path = os.path.join(BASE_DIR, edit_rel_path)

            if not os.path.exists(mask_full_path) or not os.path.exists(edit_full_path):
                print(f"Missing image files for {audit_file}: {mask_full_path} or {edit_full_path}")
                continue

            # Load images
            try:
                mask_img = Image.open(mask_full_path).convert("L")
                edit_img = Image.open(edit_full_path).convert("RGB")
                
                # Resize mask to match edit image if needed
                if mask_img.size != edit_img.size:
                    mask_img = mask_img.resize(edit_img.size, Image.NEAREST)

                # Create Ref GT (Cutout)
                # Create a black background
                ref_gt_img = Image.new("RGB", edit_img.size, (0, 0, 0))
                # Paste edit image using mask
                ref_gt_img.paste(edit_img, (0, 0), mask=mask_img)
                
                # Save Ref GT
                item_idx = audit_data.get("item_idx")
                ref_gt_filename = f"ref_gt_{item_idx}.png"
                ref_gt_full_path = os.path.join(REF_GT_DIR, ref_gt_filename)
                ref_gt_img.save(ref_gt_full_path)
                
                # Get relative path for JSON (relative to BASE_DIR, or absolute? usually relative to dataset root)
                # Assuming the training script expects paths relative to dataset_base_path
                ref_gt_rel_path = os.path.relpath(ref_gt_full_path, BASE_DIR)
                
                # Verify paths in json are relative to BASE_DIR too
                # Usually log_data paths like "openimages/source/..." are already relative to BASE_DIR if BASE_DIR is the root of the dataset structure.
                
            except Exception as e:
                print(f"Error processing images for {audit_file}: {e}")
                continue

            # 6. 构建 Prompt
            instruction = log_data.get("instruction", "")
            prompt = f"Picture 1 is the background image, the rest pictures are objects to insert. Generate a new image: {instruction}"

            # 7. Add to dataset
            entry = {
                "prompt": prompt,
                "image": bg_rel_path,
                "edit_image": [edit_rel_path],
                "item": instruction, 
                "ref_gt": ref_gt_rel_path,
                "back_mask": mask_rel_path
            }
            dataset.append(entry)

        except Exception as e:
            print(f"Error processing {audit_file}: {e}")

    # Save final JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated dataset with {len(dataset)} entries saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    process()
