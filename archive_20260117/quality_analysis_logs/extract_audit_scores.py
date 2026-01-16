import json
import os
from tqdm import tqdm

# 配置
DATASET_JSON = "dataset_qwen_pe_test.json"
AUDIT_DIR = "pico-banana-400k-subject_driven/openimages/dreamsim_mask_audit_mask"
OUTPUT_REPORT = "dataset_with_audit_scores.json"

def main():
    if not os.path.exists(DATASET_JSON):
        print(f"Error: {DATASET_JSON} not found.")
        return

    with open(DATASET_JSON, 'r') as f:
        dataset = json.load(f)
    
    print(f"正在为 {len(dataset)} 个样本提取 DreamSim 审计分数...")
    
    enriched_data = []
    missing_count = 0
    
    for item in tqdm(dataset):
        # 从 image 路径推断 item_idx
        # 例如: target_images/target_10004.png -> idx 10004
        try:
            filename = os.path.basename(item["image"])
            # 处理可能的不同命名格式，比如 target_10004.png
            parts = filename.replace('.', '_').split('_')
            item_idx = None
            for p in parts:
                if p.isdigit():
                    item_idx = p
                    break
            
            if item_idx:
                audit_path = os.path.join(AUDIT_DIR, f"item_{item_idx}_dreamsim_audit.json")
                
                if os.path.exists(audit_path):
                    with open(audit_path, 'r') as f_audit:
                        audit_data = json.load(f_audit)
                    
                    # 提取关键分数
                    global_res = audit_data.get("results", {}).get("global", {})
                    bg_dist = global_res.get("bg_dist_mean", 0.0)
                    item["audit_bg_dist"] = bg_dist
                else:
                    item["audit_bg_dist"] = None
                    missing_count += 1
            else:
                item["audit_bg_dist"] = None
                missing_count += 1
        except Exception:
            item["audit_bg_dist"] = None
            missing_count += 1
            
        enriched_data.append(item)

    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    print(f"完成！已保存至 {OUTPUT_REPORT}。缺失审计文件的样本数: {missing_count}")

if __name__ == "__main__":
    main()
