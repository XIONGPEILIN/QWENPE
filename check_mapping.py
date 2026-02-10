import os
import pandas as pd
import glob

FINAL_DIR = "final_comparison_results"
PICO_DIR = "pico_test"

def find_source_dir(filename):
    # Search for this filename in all pico_test subdirectories
    # We look for the exact image file
    matches = []
    # Get all subdirs
    subdirs = [d for d in os.listdir(PICO_DIR) if os.path.isdir(os.path.join(PICO_DIR, d))]
    
    for d in subdirs:
        candidate = os.path.join(PICO_DIR, d, filename)
        if os.path.exists(candidate):
            matches.append(d)
    
    return matches

def main():
    final_csvs = glob.glob(os.path.join(FINAL_DIR, "*.csv"))
    
    print("Investigating mappings...")
    print(f"{ 'Final CSV File':<40} | { 'Found Source Directory (in pico_test)'}")
    print("-" * 90)
    
    mapping_code = "MAPPING = {\n"
    
    for csv_path in final_csvs:
        csv_name = os.path.basename(csv_path)
        if csv_name == "all_methods_summary.csv": continue # Skip summary
        if "old" in csv_name: continue # Skip old backups

        try:
            df = pd.read_csv(csv_path)
            # Assume 'filename' column exists and has image names
            if 'filename' in df.columns and not df.empty:
                sample_file = df.iloc[0]['filename']
                found_dirs = find_source_dir(sample_file)
                
                if len(found_dirs) == 1:
                    print(f"{csv_name:<40} | {found_dirs[0]}")
                    mapping_code += f"    '{csv_name}': 'pico_test/{found_dirs[0]}',
"
                elif len(found_dirs) > 1:
                    print(f"{csv_name:<40} | AMBIGUOUS: {found_dirs}")
                    # Heuristic: match name similarity
                    best_match = None
                    max_overlap = 0
                    for d in found_dirs:
                        # Simple common substring check
                        overlap = len(set(csv_name.split('_')) & set(d.split('_')))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_match = d
                    if best_match:
                         mapping_code += f"    '{csv_name}': 'pico_test/{best_match}', # Best guess from {found_dirs}\n"
                else:
                    print(f"{csv_name:<40} | NOT FOUND")
            else:
                print(f"{csv_name:<40} | SKIPPING (No filename col or empty)")
        except Exception as e:
            print(f"{csv_name:<40} | ERROR: {e}")
            
    mapping_code += "}"
    print("\nGenerated Mapping Code for update script:\n")
    print(mapping_code)

if __name__ == "__main__":
    main()
