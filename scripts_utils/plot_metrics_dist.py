import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
RESULTS_DIR = 'final_comparison_results'
OUTPUT_DIR = 'metrics_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAMES = ['qwen_w_ste', 'qwen_wo_ste-sub', 'qwen_noste_30k', 'ace_plus', 'flux', 'magicbrush']
METRICS = ['siglip2_i_mask', 'dino_mask', 'dreamsim_mask', 'l1_out_mask']

def plot_distributions():
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

    # 2. Collect All Data
    all_data = []
    for model in MODEL_NAMES:
        csv_path = os.path.join(RESULTS_DIR, f"{model}_full.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Filter Top1000
            df = df[df['filename'].isin(file_to_type.keys())].copy()
            df['Model'] = model
            all_data.append(df)
        else:
            print(f"Warning: {csv_path} not found.")

    if not all_data:
        print("No data found.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # 3. Identify all numeric metrics
    # Drop non-numeric columns
    numeric_df = combined_df.select_dtypes(include=[np.number])
    all_metrics = [c for c in numeric_df.columns if c not in ['edit_type']]
    
    print(f"Plotting {len(all_metrics)} metrics: {all_metrics}")

    # 4. Plot Distributions (Boxplot + Violin)
    sns.set_theme(style="whitegrid")
    
    for metric in all_metrics:
        plt.figure(figsize=(14, 8))
        
        # Violin plot with inner boxplot and cut=0 to respect data bounds
        ax = sns.violinplot(x="Model", y=metric, data=combined_df, inner="box", scale="width", palette="muted", cut=0)
        
        # Add Title and Labels
        plt.title(f"Distribution of {metric} across Models (Top 1000)", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45)
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, f"dist_{metric}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot: {out_path}")

    # 5. Print Summary Stats
    print("\n--- Summary Statistics (Mean / Std) ---")
    summary = combined_df.groupby('Model')[all_metrics].agg(['mean', 'std'])
    print(summary.to_markdown(floatfmt=".4f"))

if __name__ == "__main__":
    plot_distributions()
