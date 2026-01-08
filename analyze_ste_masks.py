import os
import sys
import glob
import re
import numpy as np
from PIL import Image
from collections import defaultdict
import argparse

def analyze_masks(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Patterns
    # main_step_049_block_59.png
    pattern = re.compile(r"(main|sub)_step_(\d+)_block_(\d+)\.png")

    files = glob.glob(os.path.join(input_dir, "*.png"))
    if not files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(files)} mask files. Processing...")

    # Accumulators: key -> list of images (arrays)
    # We want to compute:
    # 1. Avg over blocks (for each step) -> Key: (type, step)
    # 2. Avg over steps (for each block) -> Key: (type, block)
    
    # To save memory, we accumulate sums and counts
    # acc[(type, 'step', index)] = {'sum': np.array, 'count': int}
    # acc[(type, 'block', index)] = {'sum': np.array, 'count': int}
    
    data = defaultdict(lambda: {'sum': None, 'count': 0})
    
    # We will determine the size for each type dynamically
    type_shapes = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        match = pattern.match(fname)
        if not match:
            continue
        
        m_type, step_str, block_str = match.groups()
        step = int(step_str)
        block = int(block_str)
        
        try:
            img = Image.open(fpath).convert('L')
            arr = np.array(img, dtype=np.float32)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        # Check shape for this specific type
        if m_type not in type_shapes:
            type_shapes[m_type] = arr.shape
            print(f"Detected size for {m_type}: {arr.shape[1]}x{arr.shape[0]}")
        
        if arr.shape != type_shapes[m_type]:
             # If size changes within same type (unexpected), skip or resize
             continue

        # Accumulate ...

        # Accumulate for Step Average (Type + Step)
        key_step = (m_type, 'step', step)
        if data[key_step]['sum'] is None:
            data[key_step]['sum'] = np.zeros_like(arr)
        data[key_step]['sum'] += arr
        data[key_step]['count'] += 1

        # Accumulate for Block Average (Type + Block)
        key_block = (m_type, 'block', block)
        if data[key_block]['sum'] is None:
            data[key_block]['sum'] = np.zeros_like(arr)
        data[key_block]['sum'] += arr
        data[key_block]['count'] += 1

        # Accumulate for Global Average (Type + Global)
        key_global = (m_type, 'global', 0)
        if data[key_global]['sum'] is None:
            data[key_global]['sum'] = np.zeros_like(arr)
        data[key_global]['sum'] += arr
        data[key_global]['count'] += 1

    print("Computing averages and saving...")

    for key, val in data.items():
        m_type, category, index = key
        
        if val['count'] == 0:
            continue
            
        avg_arr = val['sum'] / val['count']
        avg_img = Image.fromarray(avg_arr.astype(np.uint8))
        
        # Filename: avg_over_{blocks|steps}_{type}_{index}.png
        if category == 'step':
            # Avg over blocks, specific to this step
            out_name = f"avg_over_blocks_{m_type}_step_{index:03d}.png"
        elif category == 'block':
            # Avg over steps, specific to this block
            out_name = f"avg_over_steps_{m_type}_block_{index:02d}.png"
        else:
            # Global avg
            out_name = f"global_average_{m_type}.png"
            
        out_path = os.path.join(output_dir, out_name)
        avg_img.save(out_path)
        # print(f"Saved {out_path}")

    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average STE masks over steps and blocks.")
    parser.add_argument("input_dir", type=str, help="Directory containing mask images.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Output directory.")
    
    args = parser.parse_args()
    
    # Handle the specific path format if pasted directly
    input_path = args.input_dir
    # Remove trailing slash or glob pattern if present
    if input_path.endswith("/*"):
        input_path = input_path[:-2]
    elif input_path.endswith("/**"):
        input_path = input_path[:-3]
        
    analyze_masks(input_path, args.output_dir)
