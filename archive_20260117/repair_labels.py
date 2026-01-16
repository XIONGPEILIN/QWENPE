import json
import os
from collections import Counter

# Paths
METADATA_PATH = 'pico-banana-400k-subject_driven/openimages/jsonl/sft_with_local_source_image_path.jsonl'
TOP1000_PATH = 'dataset_qwen_pe_top1000.json'
TEST_PATH = 'dataset_qwen_pe_test.json'

def repair_json_file(file_path, mapping):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    fixed_count = 0
    not_found_count = 0
    
    for item in data:
        # 1. Extract the core instruction
        prefix = "Picture 1 is the image to modify. "
        prompt = item['prompt']
        if prompt.startswith(prefix):
            instruction = prompt[len(prefix):].strip()
        else:
            instruction = prompt.strip()
        
        # 2. Match with metadata
        correct_type = mapping.get(instruction)
        
        if correct_type:
            if item.get('edit_type') != correct_type:
                item['edit_type'] = correct_type
                fixed_count += 1
        else:
            not_found_count += 1

    # 3. Save corrected data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults for {file_path}:")
    print(f"  Total items: {len(data)}")
    print(f"  Labels corrected: {fixed_count}")
    print(f"  Instructions not found in metadata: {not_found_count}")
    
    # 4. Final distribution
    dist = Counter(item.get('edit_type', 'unknown') for item in data)
    print("  New Distribution:")
    for etype, count in dist.most_common():
        print(f"    - {etype}: {count}")

def main():
    print("Loading metadata from JSONL...")
    mapping = {}
    with open(METADATA_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            etype = item.get('edit_type')
            if not etype: continue
            
            # Map both 'text' and 'summarized_text' to the correct edit_type
            txt = item.get('text', '').strip()
            stxt = item.get('summarized_text', '').strip()
            
            if txt: mapping[txt] = etype
            if stxt: mapping[stxt] = etype
    
    print(f"Metadata loaded. Unique instructions indexed: {len(mapping)}")

    # Repair both files
    repair_json_file(TOP1000_PATH, mapping)
    repair_json_file(TEST_PATH, mapping)

if __name__ == "__main__":
    main()
