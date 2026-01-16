import json
import os
import pandas as pd

# Configuration
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
RESULTS_DIR = 'final_comparison_results'

def rank_accessories_clothing():
    if not os.path.exists(TOP1000_JSON):
        print(f"Error: {TOP1000_JSON} not found.")
        return

    # 1. Load Dataset Map
    with open(TOP1000_JSON, 'r') as f:
        dataset = json.load(f)
    file_to_type = {}
    for item in dataset:
        try:
            fname = os.path.basename(item['edit_image'][0])
            file_to_type[fname] = item.get('edit_type', 'unknown')
        except: pass

    # 2. Collect Data
    data = []
    for csv_file in [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv') and 'summary' not in f]:
        model_name = csv_file.replace('_full.csv', '')
        df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file))
        df['edit_type'] = df['filename'].map(file_to_type)
        
        # Filter for Accessories & Clothing
        target_types = [
            "Add/Remove/Replace Accessories (glasses, hats, jewelry, masks)",
            "Clothing edit (change color/outfit)"
        ]
        
        df_acc = df[df['edit_type'].isin(target_types)]
        df_others = df[~df['edit_type'].isin(target_types) & df['edit_type'].notna()]
        
        if len(df_acc) > 0:
            dino_acc = df_acc['dino_mask'].mean()
            dino_others = df_others['dino_mask'].mean()
            data.append({
                "Model": model_name,
                "Accessories/Clothing (DINO)": dino_acc,
                "Core Tasks (Add/Remove/Replace) (DINO)": dino_others,
                "Gap": dino_acc - dino_others
            })

    # 3. Sort and Print
    df_res = pd.DataFrame(data).sort_values("Accessories/Clothing (DINO)", ascending=False)
    
    print("\n--- Model Ranking: Accessories & Clothing (DINO Mask) ---")
    print(df_res.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    rank_accessories_clothing()
