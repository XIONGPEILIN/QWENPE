import os
import pandas as pd

DIRS = {
    "Qwen (Main)": "pico_test/qwen_results_top1000",
    "Qwen (No-STE)": "pico_test/qwen_results_noste_30k_top1000",
    "Qwen (No-Subnoise)": "pico_test/qwen_results_wosub_top1000_pixelmask_cfg4",
    "ACE++ Adaptive": "pico_test/ace_plus_results_top1000_adaptive",
    "ACE++ Standard": "pico_test/ace_plus_results_top1000",
    "Flux": "pico_test/flux_results_top1000",
    "MagicBrush Official": "pico_test/magicbrush_results_top1000"
}

def main():
    summary = []
    for name, d in DIRS.items():
        csv_path = os.path.join(d, "siglip2_qwen_eval.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            summary.append({
                "Model": name,
                "SigLIP2_I": df['siglip2_i'].mean(),
                "T_Global (Sigmoid)": df['siglip2_t_global'].mean(),
                "T_Local (Sigmoid)": df['siglip2_t_local'].mean()
            })
        else:
            print(f"Warning: {csv_path} not found.")

    if summary:
        df_sum = pd.DataFrame(summary).sort_values("T_Local (Sigmoid)", ascending=False)
        print("\n" + "="*70)
        print("NEW SIGLIP2 (SIGMOID LOGITS) EVALUATION RESULTS")
        print("="*70)
        print(df_sum.to_markdown(index=False))
        print("="*70)
    else:
        print("No results found.")

if __name__ == "__main__":
    main()

