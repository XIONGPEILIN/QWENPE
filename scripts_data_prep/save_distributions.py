import json
import os
from collections import Counter

# Paths
METADATA_PATH = 'pico-banana-400k-subject_driven/openimages/jsonl/sft_with_local_source_image_path.jsonl'
TRAIN_CROP_PATH = 'dataset_qwen_pe_train_crop.json'
TOP1000_PATH = 'dataset_qwen_pe_top1000.json'
OUTPUT_DIST_PATH = 'dataset_distributions.json'

def get_mapping():
    print("Loading metadata for mapping...")
    mapping = {}
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            etype = item.get('edit_type')
            if not etype: continue
            
            txt = item.get('text', '').strip()
            stxt = item.get('summarized_text', '').strip()
            if txt: mapping[txt] = etype
            if stxt: mapping[stxt] = etype
    return mapping

def get_distribution(file_path, mapping):
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    counts = Counter()
    for item in data:
        # For Top1000, edit_type is already there (repaired)
        # For Train_crop, we might need to map it
        etype = item.get('edit_type')
        
        if not etype or etype == 'unknown':
            prompt = item['prompt'].replace('Picture 1 is the image to modify. ', '').strip()
            etype = mapping.get(prompt, 'unknown')
        
        counts[etype] += 1
    
    return {
        "total_items": len(data),
        "distribution": dict(counts.most_common())
    }

def main():
    mapping = get_mapping()
    
    results = {
        "dataset_qwen_pe_train_crop": get_distribution(TRAIN_CROP_PATH, mapping),
        "dataset_qwen_pe_top1000": get_distribution(TOP1000_PATH, mapping)
    }
    
    with open(OUTPUT_DIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccess! Distributions written to {OUTPUT_DIST_PATH}")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
