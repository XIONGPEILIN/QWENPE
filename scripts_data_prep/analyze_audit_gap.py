import json
import os
from collections import Counter
from tqdm import tqdm

# Configuration
AUDIT_DIR = 'pico-banana-400k-subject_driven/openimages/dreamsim_mask_audit_mask'
LOG_BASE_DIR = 'pico-banana-400k-subject_driven/openimages/pico_sam_output_ALL_20251206_032609'
FIXED_JSON = 'dataset_qwen_pe_fixed.json'

def analyze_full_audit_population():
    # 1. Get list of all audit JSON files
    audit_files = [f for f in os.listdir(AUDIT_DIR) if f.endswith('.json')]
    total_audit_count = len(audit_files)
    
    # 2. Get set of items that survived in fixed.json
    with open(FIXED_JSON, 'r') as f:
        fixed_data = json.load(f)
    fixed_indices = set()
    for item in fixed_data:
        try:
            idx = item['image'].split('target_')[-1].split('.png')[0]
            fixed_indices.add(idx)
        except: pass

    # 3. Analyze distribution of ALL audits
    counts_survived = Counter()
    counts_rejected = Counter()
    
    print(f"Analyzing {total_audit_count} audit reports...")
    for f in tqdm(audit_files):
        try:
            # Extract item_idx from filename "item_X_dreamsim_audit.json"
            item_id = f.split('item_')[-1].split('_dreamsim')[0]
            log_path = os.path.join(LOG_BASE_DIR, f"item_{item_id}", f"item_{item_id}_log.json")
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as f_log:
                    log_data = json.load(f_log)
                etype = log_data.get('original_item', {}).get('edit_type', 'unknown')
                
                if item_id in fixed_indices:
                    counts_survived[etype] += 1
                else:
                    counts_rejected[etype] += 1
        except: pass

    print("\n" + "="*60)
    print(f"AUDIT POPULATION ANALYSIS (Total: {total_audit_count})")
    print("="*60)
    print(f"Survived in fixed.json: {sum(counts_survived.values())}")
    print(f"Rejected (Quality Filtered): {sum(counts_rejected.values())}")
    print("\nRejected Category Distribution:")
    for etype, count in counts_rejected.most_common():
        print(f" - {etype}: {count}")
    print("="*60)

if __name__ == "__main__":
    analyze_full_audit_population()
