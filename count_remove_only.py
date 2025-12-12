import os
import json
import glob
from tqdm import tqdm

BASE_DIR = "pico-banana-400k-subject_driven"
AUDIT_DIR = os.path.join(BASE_DIR, "openimages/dino_mask_audit")

def count_remove_only():
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_dino_audit.json"))
    print(f"Scanning {len(audit_files)} files...")

    remove_only_count = 0
    add_only_count = 0
    mixed_count = 0
    
    for audit_file in tqdm(audit_files):
        try:
            with open(audit_file, 'r') as f:
                data = json.load(f)
            
            # Check Background
            bg_sim = data.get("results", {}).get("global", {}).get("background_bbox_sim", 0)
            if bg_sim <= 0.9:
                continue

            results = data.get("results", {})
            
            # Check Add
            add_res = results.get("add", {})
            # Look for mask in global or add section
            add_path = results.get("global", {}).get("add_mask_path") or add_res.get("kind_merged_mask_path")
            has_add = bool(add_path)

            # Check Remove
            rem_res = results.get("remove", {})
            rem_path = results.get("global", {}).get("remove_mask_path") or rem_res.get("kind_merged_mask_path")
            has_remove = bool(rem_path)

            if has_remove and not has_add:
                remove_only_count += 1
            elif has_add and not has_remove:
                add_only_count += 1
            elif has_add and has_remove:
                mixed_count += 1

        except Exception:
            pass

    print(f"\n--- Statistics (BG Sim > 0.9) ---")
    print(f"Remove Only: {remove_only_count}")
    print(f"Add Only:    {add_only_count}")
    print(f"Mixed:       {mixed_count}")
    print(f"Total Valid: {remove_only_count + add_only_count + mixed_count}")

if __name__ == "__main__":
    count_remove_only()
