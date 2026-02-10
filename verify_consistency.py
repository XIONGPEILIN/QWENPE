import os
import json
import pandas as pd

# MAPPING: Final JSON -> Pico Test Dir
MAPPING = {
    "qwen_w_ste_full_summary.json": "pico_test/qwen_results_top1000",
    "ace_plus_top1000_adaptive_summary.json": "pico_test/ace_plus_results_top1000_adaptive",
    "flux_full_summary.json": "pico_test/flux_results_top1000",
    "qwen_noste_30k_full_summary.json": "pico_test/qwen_results_noste_30k_top1000"
}

FINAL_DIR = "final_comparison_results"

def main():
    print(f"{'Model File':<40} | {'Old SigLIP2_I':<15} | {'New SigLIP2_I':<15} | {'Diff'}")
    print("-" * 90)
    
    for json_name, res_dir in MAPPING.items():
        json_path = os.path.join(FINAL_DIR, json_name)
        csv_path = os.path.join(res_dir, "siglip2_qwen_eval.csv")
        
        if not os.path.exists(json_path) or not os.path.exists(csv_path):
            continue
            
        # Load Old
        with open(json_path, 'r') as f:
            old_data = json.load(f)
            old_i = old_data.get("SigLIP2_I", 0)
            
        # Load New
        df_new = pd.read_csv(csv_path)
        new_i = df_new['siglip2_i'].mean()
        
        diff = new_i - old_i
        print(f"{json_name:<40} | {old_i:<15.6f} | {new_i:<15.6f} | {diff:+.6f}")

if __name__ == "__main__":
    main()
