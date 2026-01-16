import json
import os

METADATA_JSONL = 'pico-banana-400k-subject_driven/openimages/jsonl/sft_with_local_source_image_path.jsonl'
TRAIN_JSON = 'dataset_qwen_pe_train_crop.json'

def get_distribution():
    print("Loading metadata...")
    id_to_type = {}
    with open(METADATA_JSONL, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            id_to_type[i] = item.get('edit_type', 'unknown')

    print(f"Loading {TRAIN_JSON}...")
    with open(TRAIN_JSON, 'r') as f:
        train_data = json.load(f)

    counts = {}
    for item in train_data:
        image_path = item.get('image', '')
        if 'target_' in image_path:
            try:
                # Extract idx from target_X.png
                idx = int(image_path.split('target_')[-1].split('.png')[0])
                etype = id_to_type.get(idx, 'unknown')
            except:
                etype = 'unknown'
        else:
            etype = 'unknown'
        
        counts[etype] = counts.get(etype, 0) + 1

    print("\n--- Training Set Distribution (dataset_qwen_pe_train_crop.json) ---")
    total = sum(counts.values())
    for etype, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{etype}: {count} ({count/total*100:.1f}%)")

if __name__ == "__main__":
    get_distribution()
