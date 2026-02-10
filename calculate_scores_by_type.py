import json
import csv
import os
import glob
import pandas as pd
import numpy as np

def load_edit_type_mapping(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    for item in data:
        # Assuming edit_image is a list and we take the first one
        # or we might need to handle multiple? The CSV seems to have one row per image.
        # The CSV filename matches the basename of the edit_image.
        if "edit_image" in item and item["edit_image"]:
            # item["edit_image"] is a list, e.g. ["fixed_images/fixed_10004.png"]
            for img_path in item["edit_image"]:
                filename = os.path.basename(img_path)
                mapping[filename] = item.get("edit_type", "Unknown")
    return mapping

def process_results(results_dir, mapping):
    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    all_results = {}

    for csv_file in csv_files:
        method_name = os.path.basename(csv_file).replace('.csv', '')
        print(f"Processing {method_name}...")
        
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        if 'filename' not in df.columns:
            print(f"Skipping {csv_file}: 'filename' column not found.")
            continue

        # Map filenames to edit_types
        df['edit_type'] = df['filename'].map(mapping)
        
        # Check for unmapped files
        unmapped = df[df['edit_type'].isna()]
        if not unmapped.empty:
            print(f"Warning: {len(unmapped)} rows in {csv_file} could not be mapped to an edit_type.")
            # assign 'Unknown' to unmapped
            df['edit_type'] = df['edit_type'].fillna('Unknown')

        # Calculate averages by edit_type
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Group by edit_type and calculate mean
        grouped = df.groupby('edit_type')[numeric_cols].mean()
        
        # Convert to dictionary
        all_results[method_name] = grouped.to_dict(orient='index')

    return all_results

def main():
    dataset_json = "dataset_qwen_pe_top1000_captioned.json"
    results_dir = "final_comparison_results"
    output_file = "final_comparison_by_edit_type.json"

    print("Loading edit type mapping...")
    mapping = load_edit_type_mapping(dataset_json)
    print(f"Loaded mapping for {len(mapping)} images.")

    print("Processing result files...")
    results = process_results(results_dir, mapping)

    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()
