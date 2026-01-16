import json
import os
from collections import Counter

# Paths
TEST_JSON = 'dataset_qwen_pe_test.json'
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
OUTPUT_REMAINS_JSON = 'dataset_qwen_pe_test_remains.json'

def extract_remaining_test_samples():
    if not os.path.exists(TEST_JSON) or not os.path.exists(TOP1000_JSON):
        print("Error: Missing input test files.")
        return

    # 1. Load Data
    with open(TEST_JSON, 'r') as f:
        test_data = json.load(f)
    with open(TOP1000_JSON, 'r') as f:
        top1000_data = json.load(f)

    # 2. Use 'image' path as unique key
    top1000_keys = {item['image'] for item in top1000_data}
    
    # 3. Filter out items already in Top 1000
    remains = [item for item in test_data if item['image'] not in top1000_keys]

    # 4. Save to new file
    with open(OUTPUT_REMAINS_JSON, 'w', encoding='utf-8') as f:
        json.dump(remains, f, indent=2, ensure_ascii=False)

    # 5. Statistics
    print(f"Extraction Complete:")
    print(f" - Full Test Set: {len(test_data)}")
    print(f" - Top 1000: {len(top1000_data)}")
    print(f" - Remaining (Saved): {len(remains)}")
    
    dist = Counter(item.get('edit_type', 'unknown') for item in remains)
    print("\nDistribution of Remaining 444 Samples:")
    for etype, count in dist.most_common():
        print(f"  * {etype}: {count}")

if __name__ == "__main__":
    extract_remaining_test_samples()
