import json
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = 'metrics_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_data():
    results = []
    # Pattern to match: compare/grid_search/2011-ste-28000_cfg*_alpha*/evaluation_results_summary.json
    pattern = "compare/grid_search/2011-ste-28000_cfg*_alpha*/evaluation_results_summary.json"
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} summary files.")
    
    for p in files:
        # Parse path: compare/grid_search/2011-ste-28000_cfg1_alpha0.1/evaluation_results_summary.json
        folder_name = os.path.basename(os.path.dirname(p))
        try:
            parts = folder_name.split("_")
            cfg_part = [x for x in parts if x.startswith("cfg")][0]
            alpha_part = [x for x in parts if x.startswith("alpha")][0]
            
            cfg = float(cfg_part.replace("cfg", ""))
            alpha = float(alpha_part.replace("alpha", ""))
            
            with open(p, 'r') as f:
                data = json.load(f)
                
            # Flatten dict
            row = {"CFG": cfg, "Alpha": alpha}
            row.update(data)
            results.append(row)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            
    return pd.DataFrame(results)

def plot_trends(df):
    if df.empty:
        print("No data to plot.")
        return

    # Check if we need to rename any column
    rename_map = {}
    for col in df.columns:
        if "CLIP" in col:
            rename_map[col] = col.replace("CLIP", "SigLIP2")
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"Renamed columns: {rename_map}")
        
    # Identify all numeric metric columns (excluding CFG and Alpha)
    metrics = [c for c in df.columns if c not in ["CFG", "Alpha"] and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Plotting {len(metrics)} metrics: {metrics}")
    
    sns.set_theme(style="whitegrid")
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Line plot with markers, hue by CFG
        try:
            sns.lineplot(data=df, x="Alpha", y=metric, hue="CFG", marker="o", palette="viridis")
            
            plt.title(f"Effect of Alpha on {metric}", fontsize=16)
            plt.xlabel("Inpaint Blend Alpha", fontsize=14)
            plt.ylabel(metric, fontsize=14)
            
            out_path = os.path.join(OUTPUT_DIR, f"alpha_trend_{metric}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Could not plot {metric}: {e}")
            plt.close()

if __name__ == "__main__":
    df = collect_data()
    if not df.empty:
        df = df.sort_values(by=["CFG", "Alpha"])
        print(df[["CFG", "Alpha", "MAE", "MAE_OutMask"]].to_markdown(index=False, floatfmt=".4f"))
        plot_trends(df)
