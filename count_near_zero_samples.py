import os
import pandas as pd

# MAPPING: Model Name -> New CSV Path
MAPPING = {
    "Qwen (Main)": "pico_test/qwen_results_top1000/siglip2_qwen_eval.csv",
    "Qwen (No-STE)": "pico_test/qwen_results_noste_30k_top1000/siglip2_qwen_eval.csv",
    "Qwen (No-Sub)": "pico_test/qwen_results_wosub_top1000_pixelmask_cfg4/siglip2_qwen_eval.csv",
    "ACE++ Adap": "pico_test/ace_plus_results_top1000_adaptive/siglip2_qwen_eval.csv",
    "ACE++ Std": "pico_test/ace_plus_results_top1000/siglip2_qwen_eval.csv",
    "Flux": "pico_test/flux_results_top1000/siglip2_qwen_eval.csv",
    "MagicBrush": "pico_test/magicbrush_results_top1000/siglip2_qwen_eval.csv"
}

THRESHOLD = 0.01 # Scores below this are considered "near zero" failure cases

def main():
    print(f"Counting failure cases (SigLIP2 T Score < {THRESHOLD})...\n")
    results = []
    
    for name, csv_path in MAPPING.items():
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        total = len(df)
        
        # Count exactly 0 (likely Mask error)
        exact_zero_local = (df['siglip2_t_local'] == 0).sum()
        
        # Count near zero failures
        fail_global = (df['siglip2_t_global'] < THRESHOLD).sum()
        fail_local = (df['siglip2_t_local'] < THRESHOLD).sum()
        
        results.append({
            "Model": name,
            "Total": total,
            "Exact 0 Local": exact_zero_local,
            "Fail Global (<0.01)": fail_global,
            "Fail Local (<0.01)": fail_local
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False))
    
    print("\nNote: 'Exact 0 Local' usually means the BBox/Mask extraction failed for that sample.")

if __name__ == "__main__":
    main()
