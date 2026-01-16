import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = 'metrics_plots/grid_dist'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_grid_csvs():
    pattern = "compare/grid_search/*/evaluation_results.csv"
    files = glob.glob(pattern)
    print(f"Found {len(files)} CSV files.")
    
    all_dfs = []
    for p in files:
        folder_name = os.path.basename(os.path.dirname(p))
        try:
            parts = folder_name.split("_")
            cfg_part = [x for x in parts if x.startswith("cfg")][0]
            alpha_part = [x for x in parts if x.startswith("alpha")][0]
            
            cfg = float(cfg_part.replace("cfg", ""))
            alpha = float(alpha_part.replace("alpha", ""))
            
            df = pd.read_csv(p)
            df["CFG"] = cfg
            df["Alpha"] = alpha
            all_dfs.append(df)
        except Exception as e:
            print(f"Error parsing {p}: {e}")
            
    if not all_dfs:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def plot_distributions(df):
    if df.empty:
        print("No data.")
        return

    # Clean rename
    col_map = {}
    for c in df.columns:
        if c in ["filename", "prompt", "CFG", "Alpha"]: continue
        
        new_c = c
        new_c = new_c.replace("clip", "SigLIP2")
        new_c = new_c.replace("dino", "DINO")
        new_c = new_c.replace("dreamsim", "DS")
        new_c = new_c.replace("l1", "MAE")
        new_c = new_c.replace("l2", "MSE")
        # Fix casing
        new_c = new_c.replace("_i", "_I").replace("_t", "_T")
        new_c = new_c.replace("_bbox", "_BBox").replace("_mask", "_Mask")
        new_c = new_c.replace("_in_", "_In").replace("_out_", "_Out")
        col_map[c] = new_c
    
    df = df.rename(columns=col_map)
    
    # Identify numeric metrics
    metrics = [c for c in df.columns if c not in ["filename", "prompt", "CFG", "Alpha"] and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Plotting {len(metrics)} metrics...")
    
    sns.set_theme(style="whitegrid")
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        try:
            ax = sns.violinplot(
                data=df, 
                x="Alpha", 
                y=metric, 
                hue="CFG", 
                inner="box", 
                palette="viridis",
                cut=0
            )
            
            plt.title(f"Distribution of {metric} by Alpha & CFG", fontsize=16)
            plt.xlabel("Alpha", fontsize=14)
            plt.ylabel(metric, fontsize=14)
            
            out_path = os.path.join(OUTPUT_DIR, f"dist_{metric}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed to plot {metric}: {e}")
            plt.close()

if __name__ == "__main__":
    df = load_all_grid_csvs()
    plot_distributions(df)
