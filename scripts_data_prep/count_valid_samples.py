import os
import json
import glob
from tqdm import tqdm

BASE_DIR = "pico-banana-400k-subject_driven"
AUDIT_DIR = os.path.join(BASE_DIR, "openimages/dino_mask_audit")

def count():
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_dino_audit.json"))
    print(f"Total audit files: {len(audit_files)}")

    count_valid = 0
    count_high_bg = 0
    count_high_sub = 0

    for audit_file in tqdm(audit_files):
        try:
            with open(audit_file, 'r') as f:
                audit_data = json.load(f)
            
            results = audit_data.get("results", {})
            global_res = results.get("global", {})
            
            # 1. Check Background Sim
            bg_sim = global_res.get("background_bbox_sim", 0)
            if bg_sim > 0.9:
                count_high_bg += 1
            else:
                continue # Skip if bg not good enough

            # 2. Check Sub Mask Sim (in 'add' section)
            add_res = results.get("add")
            if not add_res:
                continue # Not an add task

            sub_mask_results = add_res.get("sub_mask_results", [])
            if not sub_mask_results:
                continue

            # Check if ANY sub mask has high similarity (or ALL? usually ANY is good enough for existence)
            # Let's check if at least one object is well detected
            max_sub_sim = 0
            for sub in sub_mask_results:
                sim = sub.get("cos_sim", 0)
                if sim > max_sub_sim:
                    max_sub_sim = sim
            
            if max_sub_sim > 0.9:
                count_high_sub += 1
                count_valid += 1

        except Exception as e:
            pass

    print(f"\n--- Statistics ---")
    print(f"Total Files: {len(audit_files)}")
    print(f"High BG Sim (>0.9): {count_high_bg}")
    print(f"High Sub Sim (>0.9): {count_high_sub} (Given High BG)")
    print(f"Total Valid: {count_valid}")

if __name__ == "__main__":
    count()
