import json
import os
from collections import Counter
from tqdm import tqdm

# Configuration
LOG_BASE_DIR = 'pico-banana-400k-subject_driven/openimages/pico_sam_output_ALL_20251206_032609'
DATASETS = {
    "dataset_qwen_pe_train_crop": 'dataset_qwen_pe_train_crop.json',
    "dataset_qwen_pe_top1000": 'dataset_qwen_pe_top1000.json',
    "dataset_qwen_pe_test_remains": 'dataset_qwen_pe_test_remains.json'
}
OUTPUT_DIST_PATH = 'dataset_distributions.json'

def get_accurate_distribution(file_path):
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    counts = Counter()
    print(f"Processing {file_path} using log-based lookup...")
    
    for item in tqdm(data):
        # 1. Extract Item ID from image path (e.g., target_images/target_7095.png)
        img_path = item.get('image', '')
        try:
            item_id = img_path.split('target_')[-1].split('.png')[0]
            # 2. Construct log path
            log_path = os.path.join(LOG_BASE_DIR, f"item_{item_id}", f"item_{item_id}_log.json")
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as f_log:
                    log_data = json.load(f_log)
                
                # 3. Extract the definitive edit_type from original_item in log
                etype = log_data.get('original_item', {}).get('edit_type', 'unknown')
                counts[etype] += 1
            else:
                counts['Log Missing'] += 1
        except Exception:
            counts['Parsing Error'] += 1
            
    return {
        "total_items": len(data),
        "distribution": dict(counts.most_common())
    }

def main():
    results = {}
    for key, path in DATASETS.items():
        results[key] = get_accurate_distribution(path)
    
    with open(OUTPUT_DIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinal Accurate Distributions written to {OUTPUT_DIST_PATH}")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
