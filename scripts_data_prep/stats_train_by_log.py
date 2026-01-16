import json
import os
from collections import Counter
from tqdm import tqdm

# Configuration
TRAIN_JSON = 'dataset_qwen_pe_top1000.json'
LOG_BASE_DIR = 'pico-banana-400k-subject_driven/openimages/pico_sam_output_ALL_20251206_032609'

def get_train_dist_by_log():
    if not os.path.exists(TRAIN_JSON):
        print(f"Error: {TRAIN_JSON} not found.")
        return

    with open(TRAIN_JSON, 'r') as f:
        data = json.load(f)

    counts = Counter()
    missing_logs = 0
    
    print(f"Analyzing {len(data)} items...")
    
    for item in tqdm(data):
        # Extract ID from "target_images/target_7095.png"
        img_path = item.get('image', '')
        try:
            item_id = img_path.split('target_')[-1].split('.png')[0]
            log_path = os.path.join(LOG_BASE_DIR, f"item_{item_id}", f"item_{item_id}_log.json")
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Get edit_type from original_item
                etype = log_data.get('original_item', {}).get('edit_type', 'unknown')
                counts[etype] += 1
            else:
                missing_logs += 1
                counts['Log Missing'] += 1
        except Exception:
            counts['Error Parsing ID'] += 1

    print("\n" + "="*50)
    print(f"DISTRIBUTION FOR: {TRAIN_JSON}")
    print("="*50)
    total = sum(counts.values())
    for etype, count in counts.most_common():
        print(f"{etype}: {count} ({count/total*100:.1f}%)")
    print("="*50)
    print(f"Total processed: {total}")
    print(f"Missing logs: {missing_logs}")

if __name__ == "__main__":
    get_train_dist_by_log()
