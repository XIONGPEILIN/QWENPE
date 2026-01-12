import json
import os
import glob
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = "pico-banana-400k-subject_driven"
AUDIT_DIR = os.path.join(BASE_DIR, "openimages/dreamsim_mask_audit_mask")
JSON_PATH = "dataset_qwen_pe_fixed_updated.json"

def patch_prompts():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        dataset = json.load(f)

    # 1. Map item_idx to summarized_text
    print("Loading audit and log files to fetch short prompts...")
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*.json"))
    item_to_prompt = {}
    
    for af in tqdm(audit_files):
        try:
            with open(af, 'r') as f:
                audit_data = json.load(f)
            
            item_idx = str(audit_data.get("item_idx"))
            log_rel_path = audit_data.get("log_path")
            
            if log_rel_path:
                log_full_path = os.path.join(BASE_DIR, log_rel_path)
                if os.path.exists(log_full_path):
                    with open(log_full_path, 'r') as f_log:
                        log_data = json.load(f_log)
                    # Extract summarized_text from original_item
                    short_prompt = log_data.get("original_item", {}).get("summarized_text", "")
                    if not short_prompt:
                        short_prompt = log_data.get("instruction", "")
                    
                    item_to_prompt[item_idx] = short_prompt
        except Exception:
            continue

    # 2. Update prompts in dataset
    print("Patching JSON entries...")
    updated_count = 0
    for entry in dataset:
        img_name = os.path.basename(entry["image"])
        item_idx = img_name.replace("target_", "").replace(".png", "")
        
        if item_idx in item_to_prompt:
            entry["prompt"] = f"Picture 1 is the image to modify. {item_to_prompt[item_idx]}"
            updated_count += 1

    # 3. Save back
    with open(JSON_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Successfully updated {updated_count} prompts in {JSON_PATH}")

if __name__ == "__main__":
    patch_prompts()
