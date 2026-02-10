import json
import pandas as pd
from tabulate import tabulate

def main():
    json_path = "final_comparison_by_edit_type.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # We want to compare SigLIP2_T_Global across methods and edit types
    metric_name = "SigLIP2_T_Global"
    
    # Structure:
    # Rows: Edit Types
    # Columns: Methods
    
    edit_types = set()
    methods = list(data.keys())
    
    # Collect all edit types
    for method in methods:
        for edit_type in data[method]:
            edit_types.add(edit_type)
            
    sorted_edit_types = sorted(list(edit_types))
    
    # Prepare data for table
    table_data = []
    
    for edit_type in sorted_edit_types:
        row = [edit_type]
        for method in methods:
            val = data[method].get(edit_type, {}).get(metric_name, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(val)
        table_data.append(row)
        
    headers = ["Edit Type"] + methods
    
    print(f"Comparison for metric: {metric_name}\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Also save to a CSV for easier copy-pasting if needed
    df = pd.DataFrame(table_data, columns=headers)
    csv_filename = "comparison_SigLIP2_T_Global.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved comparison table to {csv_filename}")

if __name__ == "__main__":
    main()

