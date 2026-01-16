import json
import os
import pandas as pd

# Configuration
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
RESULTS_DIR = 'final_comparison_results'

def rank_detailed_metrics():
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
    
    target_types = {
        "acc": "Add/Remove/Replace Accessories (glasses, hats, jewelry, masks)",
        "cloth": "Clothing edit (change color/outfit)"
    }

    for csv_file in [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv') and 'summary' not in f]:
        model_name = csv_file.replace('_full.csv', '')
        df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file))
        df['edit_type'] = df['filename'].map(file_to_type)
        
        entry = {"Model": model_name}
        
        for short_name, full_type in target_types.items():
            sub_df = df[df['edit_type'] == full_type]
            if len(sub_df) > 0:
                # Mask Metrics
                entry[f"{short_name}_SigLIP_Mask"] = sub_df.get('siglip2_i_mask', pd.Series([0])).mean()
                entry[f"{short_name}_DINO_Mask"] = sub_df.get('dino_mask', pd.Series([0])).mean()
                entry[f"{short_name}_DreamSim_Mask"] = sub_df.get('dreamsim_mask', pd.Series([0])).mean()
                
                # Full Image Metrics
                entry[f"{short_name}_SigLIP_Full"] = sub_df.get('siglip2_i', pd.Series([0])).mean()
                entry[f"{short_name}_DINO_Full"] = sub_df.get('dino', pd.Series([0])).mean()
                entry[f"{short_name}_DreamSim_Full"] = sub_df.get('dreamsim', pd.Series([0])).mean()
        
        data.append(entry)

    df_res = pd.DataFrame(data)

    # 3. Print Tables
    print_rankings(df_res, "acc", "ACCESSORIES")
    print_rankings(df_res, "cloth", "CLOTHING")

def print_rankings(df, prefix, title):
    print(f"\n{'='*20} {title} RANKINGS {'='*20}")
    
    # SigLIP (Higher is better)
    print(f"\n--- {title}: SigLIP (Semantic) [Higher is Better] ---")
    view = df[["Model", f"{prefix}_SigLIP_Mask", f"{prefix}_SigLIP_Full"]].sort_values(f"{prefix}_SigLIP_Mask", ascending=False)
    print(view.to_markdown(index=False, floatfmt=".4f"))

    # DINO (Higher is better)
    print(f"\n--- {title}: DINO (Structure) [Higher is Better] ---")
    view = df[["Model", f"{prefix}_DINO_Mask", f"{prefix}_DINO_Full"]].sort_values(f"{prefix}_DINO_Mask", ascending=False)
    print(view.to_markdown(index=False, floatfmt=".4f"))

    # DreamSim (Lower is better usually, BUT for editing tasks, moderate distance means change happened)
    # Here we just list it. For 'Mask', lower means 'natural integration' or 'less change'? 
    # Actually for editing, we want Semantic match. DreamSim measures visual distance.
    # We will sort by Mask Distance (Ascending) just for consistency, but interpret with caution.
    print(f"\n--- {title}: DreamSim (Perceptual Distance) [Lower means 'Less Change'] ---")
    view = df[["Model", f"{prefix}_DreamSim_Mask", f"{prefix}_DreamSim_Full"]].sort_values(f"{prefix}_DreamSim_Mask", ascending=True)
    print(view.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    rank_detailed_metrics()
