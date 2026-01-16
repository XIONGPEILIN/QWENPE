import json
from collections import Counter

# Configuration
RANKING_JSON = "final_quality_ranking.json"
DATASET_JSON = "dataset_qwen_pe_test.json"
OUTPUT_JSON = "dataset_qwen_pe_top1000.json"
KEEP_COUNT = 1000

def main():
    # 1. Load Ranking & Sort
    with open(RANKING_JSON, 'r') as f:
        ranking = json.load(f)
    
    # Sort by Final Score ASCENDING (Lower score = Better quality)
    ranking.sort(key=lambda x: x["final_score"])
    
    # Keep Top N (Best quality)
    top_entries = ranking[:KEEP_COUNT]
    top_indices = {entry["index"] for entry in top_entries}
    
    print(f"Total Ranked Samples: {len(ranking)}")
    print(f"Keeping Top: {len(top_entries)} (Score Range: {top_entries[0]['final_score']:.4f} to {top_entries[-1]['final_score']:.4f})")

    # 2. Load Full Dataset to get Edit Types
    with open(DATASET_JSON, 'r') as f:
        full_dataset = json.load(f)
        
    filtered_dataset = []
    edit_types = []
    
    for i, item in enumerate(full_dataset):
        if i in top_indices:
            filtered_dataset.append(item)
            # Handle case where edit_type might be missing
            e_type = item.get("edit_type", "unknown")
            edit_types.append(e_type)
            
    # 3. Analyze Distribution
    distribution = Counter(edit_types)
    total = len(edit_types)
    
    print("\n--- Edit Type Distribution (Top 1000 Quality Samples) ---")
    for etype, count in distribution.most_common():
        percentage = (count / total) * 100
        print(f"{etype}: {count} ({percentage:.1f}%)")
        
    # 4. Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(filtered_dataset, f, indent=2)
    print(f"\nSaved filtered dataset to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
