import json
import os
import pandas as pd

# Configuration
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
RESULTS_DIR = 'final_comparison_results'

def rank_split_acc_clothing():
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
    acc_data = []
    cloth_data = []
    
    acc_type = "Add/Remove/Replace Accessories (glasses, hats, jewelry, masks)"
    cloth_type = "Clothing edit (change color/outfit)"

    for csv_file in [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv') and 'summary' not in f]:
        model_name = csv_file.replace('_full.csv', '')
        df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file))
        df['edit_type'] = df['filename'].map(file_to_type)
        
        # Calculate for Accessories
        df_acc = df[df['edit_type'] == acc_type]
        if len(df_acc) > 0:
            acc_data.append({"Model": model_name, "DINO (Accessories)": df_acc['dino_mask'].mean()})
            
        # Calculate for Clothing
        df_cloth = df[df['edit_type'] == cloth_type]
        if len(df_cloth) > 0:
            cloth_data.append({"Model": model_name, "DINO (Clothing)": df_cloth['dino_mask'].mean()})

    # 3. Sort and Print
    df_acc_res = pd.DataFrame(acc_data).sort_values("DINO (Accessories)", ascending=False)
    df_cloth_res = pd.DataFrame(cloth_data).sort_values("DINO (Clothing)", ascending=False)
    
    print("\n--- Model Ranking: ACCESSORIES (DINO Mask) ---")
    print(df_acc_res.to_markdown(index=False, floatfmt=".4f"))
    
    print("\n--- Model Ranking: CLOTHING EDIT (DINO Mask) ---")
    print(df_cloth_res.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    rank_split_acc_clothing()
