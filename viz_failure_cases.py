import os
import pandas as pd
import json
import cv2
import numpy as np

# Configuration
CSV_PATH = "pico_test/qwen_results_top1000/siglip2_qwen_eval.csv"
PRED_DIR = "pico_test/qwen_results_top1000"
DATASET_JSON = "dataset_qwen_pe_top1000_captioned.json"
OUTPUT_IMG = "failure_cases_full_analysis.jpg"

def draw_multiline_text(img, text_list, x, y, font_scale=0.35, thickness=1):
    """Draws multiple lines of text with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    curr_y = y
    for label, text in text_list:
        full_text = f"{label}: {text}"
        # Wrap long text
        words = full_text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if w > img.shape[1] - 20:
                lines.append(current_line)
                current_line = word + " "
            else:
                current_line = test_line
        lines.append(current_line)

        for line in lines:
            (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(img, (x, curr_y - h - 5), (x + w, curr_y + baseline), (0, 0, 0), -1)
            cv2.putText(img, line, (x, curr_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            curr_y += h + 10
        curr_y += 5 # extra space between items
    return img

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
    
    # Load data
    df = pd.read_csv(CSV_PATH)
    with open(DATASET_JSON, 'r') as f:
        dataset = json.load(f)
    
    info_map = {}
    for item in dataset:
        try:
            fname = os.path.basename(item['edit_image'][0])
            info_map[fname] = {
                "instr": item.get("prompt", "N/A"),
                "g_cap": item.get("global_caption", "N/A"),
                "l_cap": item.get("local_caption", "N/A")
            }
        except: pass

    # Get Failure Samples
    worst_local = df.sort_values('siglip2_t_local').head(10)
    worst_global = df.sort_values('siglip2_t_global').head(10)

    case_images = []
    for title, sub_df, score_key in [("WORST LOCAL", worst_local, "siglip2_t_local"), 
                                     ("WORST GLOBAL", worst_global, "siglip2_t_global")]:
        for _, row in sub_df.iterrows():
            fname = row['filename']
            score = row[score_key]
            img_path = os.path.join(PRED_DIR, fname)
            
            if not os.path.exists(img_path):
                img = np.zeros((800, 800, 3), dtype=np.uint8)
            else:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (800, 800)) # Larger size for more text
            
            info = info_map.get(fname, {"instr": "N/A", "g_cap": "N/A", "l_cap": "N/A"})
            
            # Prepare Text list
            text_data = [
                ("TYPE", title),
                ("SCORE", f"{score:.6f}"),
                ("INSTR", info["instr"]),
                ("G_CAP", info["g_cap"]),
                ("L_CAP", info["l_cap"])
            ]
            
            img = draw_multiline_text(img, text_data, 10, 30)
            case_images.append(img)

    # Combine into grid (4 rows, 5 cols)
    rows = []
    for i in range(0, 20, 5):
        rows.append(np.hstack(case_images[i:i+5]))
    
    final_grid = np.vstack(rows)
    cv2.imwrite(OUTPUT_IMG, final_grid)
    print(f"Detailed failure analysis saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
