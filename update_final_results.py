import os
import pandas as pd
import json
import shutil

# Explicit Mapping based on user confirmation
MAPPING = {
    "ace_plus_full.csv": "pico_test/ace_plus_results_top1000",
    "ace_plus_top1000_adaptive.csv": "pico_test/ace_plus_results_top1000_adaptive",
    "flux_full.csv": "pico_test/flux_results_top1000",
    "magicbrush_full.csv": "pico_test/magicbrush_results_top1000",
    "qwen_noste_30k_full.csv": "pico_test/qwen_results_noste_30k_top1000",
    "qwen_w_ste_full.csv": "pico_test/qwen_results_top1000",
    "qwen_wo_ste-sub_full.csv": "pico_test/qwen_results_wosub_top1000_pixelmask_cfg4"
}

FINAL_DIR = "final_comparison_results"

def main():
    print(f"Updating results in '{FINAL_DIR}' with new SigLIP2 Text metrics...\n")
    
    for csv_name, res_dir in MAPPING.items():
        final_csv_path = os.path.join(FINAL_DIR, csv_name)
        new_metrics_path = os.path.join(res_dir, "siglip2_qwen_eval.csv")
        
        if not os.path.exists(final_csv_path):
            print(f"[SKIP] Final CSV not found: {final_csv_path}")
            continue
            
        if not os.path.exists(new_metrics_path):
            print(f"[SKIP] New metrics not found for {res_dir}")
            continue
            
        print(f"Processing {csv_name} (from {res_dir})...")
        
        # Backup original
        shutil.copy2(final_csv_path, final_csv_path + ".bak")
        
        # Load Dataframes
        df_final = pd.read_csv(final_csv_path)
        df_new = pd.read_csv(new_metrics_path)
        
        # Key column for merging
        key_col = 'filename'
        if key_col not in df_final.columns:
            if 'image' in df_final.columns:
                key_col = 'image'
            else:
                # Fallback to order-based alignment if count matches
                if len(df_final) == len(df_new):
                    print(f"  Warning: No common key column found. Using index alignment.")
                    key_col = None
                else:
                    print(f"  Error: Cannot align data. Check key columns.")
                    continue

        # Prepare new columns
        # siglip2_t_global -> SigLIP2_T_Global
        # siglip2_t_local -> SigLIP2_T_Local
        subset_new = df_new[['filename', 'siglip2_t_global', 'siglip2_t_local']].copy()
        subset_new.rename(columns={
            'filename': key_col if key_col else 'filename',
            'siglip2_t_global': 'SigLIP2_T_Global',
            'siglip2_t_local': 'SigLIP2_T_Local'
        }, inplace=True)
        
        # Remove old columns if they exist to prevent duplicates
        for col in ['SigLIP2_T_Global', 'SigLIP2_T_Local']:
            if col in df_final.columns:
                df_final.drop(columns=[col], inplace=True)
        
        # Merge
        if key_col:
            df_merged = pd.merge(df_final, subset_new, on=key_col, how='left')
        else:
            df_merged = df_final.copy()
            df_merged['SigLIP2_T_Global'] = subset_new['SigLIP2_T_Global']
            df_merged['SigLIP2_T_Local'] = subset_new['SigLIP2_T_Local']
            
        # Save Updated CSV
        df_merged.to_csv(final_csv_path, index=False)
        
        # Update JSON Summary
        json_path = os.path.join(FINAL_DIR, os.path.splitext(csv_name)[0] + "_summary.json")
        numeric_cols = df_merged.select_dtypes(include=[float, int]).columns
        summary = df_merged[numeric_cols].mean().to_dict()
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"  Successfully updated CSV and JSON.")

    print("\nAll updates complete!")

if __name__ == "__main__":
    main()