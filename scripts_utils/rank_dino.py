import json
import os
import pandas as pd

# Configuration
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
RESULTS_DIR = 'final_comparison_results'

def rank_models_by_dino():
    if not os.path.exists(TOP1000_JSON):
        print(f"Error: {TOP1000_JSON} not found. Please run repair_labels.py first.")
        return

    # 1. Load Dataset and Map Filename -> Edit Type
    with open(TOP1000_JSON, 'r') as f:
        dataset = json.load(f)
    
    file_to_type = {}
    for item in dataset:
        try:
            fname = os.path.basename(item['edit_image'][0])
            file_to_type[fname] = item.get('edit_type', 'unknown')
        except: pass

    # 2. Process CSVs
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv') and 'summary' not in f]
    
    rankings = []

    for csv_file in csv_files:
        csv_path = os.path.join(RESULTS_DIR, csv_file)
        model_name = csv_file.replace('_full.csv', '')
        
        try:
            df = pd.read_csv(csv_path)
            df['edit_type'] = df['filename'].map(file_to_type)
            df_filtered = df[df['edit_type'].notna()]
            
            if len(df_filtered) == 0: continue

            # Get overall DINO score
            dino_score = df_filtered['dino_mask'].mean()
            rankings.append((model_name, dino_score))

        except Exception: pass

    # 3. Sort and Print
    rankings.sort(key=lambda x: x[1], reverse=True) # Higher is better for DINO

    print("\n--- Model Ranking by DINO Mask Score (Higher is Better) ---")
    print("| Rank | Model | DINO Mask Score |")
    print("|:---:|:---|:---:|")
    for i, (name, score) in enumerate(rankings):
        print(f"| {i+1} | {name} | {score:.4f} |")

if __name__ == "__main__":
    rank_models_by_dino()
