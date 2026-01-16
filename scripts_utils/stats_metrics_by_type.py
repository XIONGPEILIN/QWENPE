import json
import os
import pandas as pd
import numpy as np

# Configuration
# Path to the repaired Top 1000 dataset (which now has correct edit_types)
TOP1000_JSON = 'dataset_qwen_pe_top1000.json'
RESULTS_DIR = 'final_comparison_results'

def calculate_metrics_by_type():
    if not os.path.exists(TOP1000_JSON):
        print(f"Error: {TOP1000_JSON} not found. Please run repair_labels.py first.")
        return

    # 1. Load Dataset and Map Filename -> Edit Type
    print(f"Loading {TOP1000_JSON}...")
    with open(TOP1000_JSON, 'r') as f:
        dataset = json.load(f)
    
    file_to_type = {}
    for item in dataset:
        # Extract filename "fixed_10004.png" from "edit_image": ["fixed_images/fixed_10004.png"]
        try:
            fname = os.path.basename(item['edit_image'][0])
            file_to_type[fname] = item.get('edit_type', 'unknown')
        except: pass
        
    print(f"Loaded {len(file_to_type)} labels.")

    # 2. Process each CSV in results dir
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv') and 'summary' not in f]
    
    all_summaries = {}

    for csv_file in csv_files:
        csv_path = os.path.join(RESULTS_DIR, csv_file)
        model_name = csv_file.replace('_full.csv', '')
        print(f"\nProcessing {model_name}...")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Add 'edit_type' column
            df['edit_type'] = df['filename'].map(file_to_type)
            
            # Filter only rows that are in our Top 1000 set
            df_filtered = df[df['edit_type'].notna()]
            
            if len(df_filtered) == 0:
                print(f"  Warning: No matching filenames found in {csv_file}")
                continue

            # Group by Edit Type and Calculate Mean
            # Define metrics to aggregate
            metrics = ['siglip2_i', 'siglip2_t', 'dino', 'dreamsim', 
                       'siglip2_i_bbox', 'dino_bbox', 'dreamsim_bbox', 
                       'siglip2_i_mask', 'dino_mask', 'dreamsim_mask',
                       'l2', 'l1', 'l2_in_mask', 'l1_in_mask', 'l2_out_mask', 'l1_out_mask']
            
            # Only use metrics present in the CSV
            valid_metrics = [m for m in metrics if m in df.columns]
            
            grouped = df_filtered.groupby('edit_type')[valid_metrics].mean()
            
            # Add 'Overall' row
            overall = df_filtered[valid_metrics].mean().to_frame().T
            overall.index = ['Overall']
            
            final_stats = pd.concat([grouped, overall])
            
            # Save or Print
            print(f"  --- {model_name} Statistics ---")
            print(final_stats[['siglip2_i_mask', 'dino_mask', 'dreamsim_mask', 'l1_out_mask']].to_markdown())
            
            all_summaries[model_name] = final_stats

        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")

    # Optional: Save consolidated report to JSON/CSV if needed
    # ...

if __name__ == "__main__":
    calculate_metrics_by_type()
