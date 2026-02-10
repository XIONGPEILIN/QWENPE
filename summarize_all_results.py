import os
import pandas as pd
import json

DIRS = [
    "pico_test/ace_plus_results_top1000_adaptive",
    "pico_test/flux_results_top1000",
    "pico_test/qwen_results_top1000",
    "pico_test/qwen_results_noste_30k_top1000",
    "pico_test/qwen_results_wosub_top1000_pixelmask_cfg4",
    "pico_test/ace_plus_results_top1000",
    "pico_test/magicbrush_results_top1000"
]

def main():
    summary_data = []
    
    for d in DIRS:
        csv_path = os.path.join(d, "siglip2_qwen_eval.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            summary_data.append({
                "Model": os.path.basename(d),
                "SigLIP2_I": df['siglip2_i'].mean(),
                "SigLIP2_T_Global": df['siglip2_t_global'].mean(),
                "SigLIP2_T_Local": df['siglip2_t_local'].mean()
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Sort by Local T-Score (usually the most important for edit precision)
        summary_df = summary_df.sort_values("SigLIP2_T_Local", ascending=False)
        
        output_path = "final_comparison_results/qwen_caption_evaluation_summary.csv"
        os.makedirs("final_comparison_results", exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        
        print("\n" + "="*60)
        print("FINAL COMPARISON SUMMARY (Based on Qwen Captions)")
        print("="*60)
        print(summary_df.to_markdown(index=False))
        print(f"\nSaved summary to: {output_path}")
    else:
        print("No evaluation results found to summarize.")

if __name__ == "__main__":
    main()
