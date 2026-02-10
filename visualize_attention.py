import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from pathlib import Path
import argparse

def visualize_attention_maps(
    attn_dir, 
    output_dir, 
    token_index=213,
    # Fallback defaults
    default_height=52, 
    default_width=78
):
    attn_dir = Path(attn_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try to load metadata
    metadata_path = attn_dir / "segments_metadata.pt"
    segments = None
    
    if metadata_path.exists():
        print(f"Loading metadata from {metadata_path}")
        try:
            segments = torch.load(metadata_path, map_location="cpu")
            print("Segments found:", segments.keys())
        except Exception as e:
            print(f"Error loading metadata: {e}")
    else:
        print("Metadata not found. Using fallback defaults for Main Image only.")

    # 2. Find files
    files = list(attn_dir.glob("*_img2txt.pt"))
    if not files:
        print(f"No *_img2txt.pt files found in {attn_dir}")
        return

    print(f"Processing {len(files)} files for token index {token_index}...")

    for pt_file in tqdm(files):
        try:
            # Parse filename
            name_parts = pt_file.stem.split('_')
            step_str = [p for p in name_parts if p.startswith('s')][0]
            layer_str = [p for p in name_parts if p.startswith('l')][0]
            
            # Load Tensor [B, Seq_Img, Seq_Txt] -> [Seq_Img, Seq_Txt]
            attn_tensor = torch.load(pt_file, map_location="cpu").float()
            if attn_tensor.dim() == 3:
                attn_tensor = attn_tensor.squeeze(0)
            
            # Check Token Index
            if token_index >= attn_tensor.shape[1]:
                continue # Skip if out of bounds

            # Extract Column [Seq_Img]
            col_attn = attn_tensor[:, token_index]

            # 3. Visualize Segments
            if segments:
                # Use Metadata
                for seg_name, info in segments.items():
                    start = info["start"]
                    end = info["end"]
                    h, w = info["h"], info["w"]
                    
                    if end > col_attn.shape[0]:
                        # print(f"Warning: Segment {seg_name} end {end} > tensor size {col_attn.shape[0]}")
                        continue
                        
                    seg_data = col_attn[start:end]
                    
                    # Reshape
                    if seg_data.numel() != h * w:
                        print(f"Error: Segment {seg_name} size {seg_data.numel()} != {h}*{w}")
                        continue
                        
                    attn_2d = seg_data.reshape(h, w).numpy()
                    save_viz(attn_2d, output_dir, step_str, layer_str, token_index, suffix=seg_name)
            else:
                # Fallback: Main Image Only
                main_len = default_height * default_width
                if col_attn.shape[0] >= main_len:
                    seg_data = col_attn[:main_len]
                    attn_2d = seg_data.reshape(default_height, default_width).numpy()
                    save_viz(attn_2d, output_dir, step_str, layer_str, token_index, suffix="main_fallback")

        except Exception as e:
            print(f"Failed to process {pt_file}: {e}")

def save_viz(attn_2d, output_dir, step, layer, token, suffix):
    # Normalize
    min_val = attn_2d.min()
    max_val = attn_2d.max()
    if max_val - min_val > 1e-6:
        attn_norm = (attn_2d - min_val) / (max_val - min_val)
    else:
        attn_norm = np.zeros_like(attn_2d)

    plt.figure(figsize=(4, 4))
    plt.imshow(attn_norm, cmap='magma', interpolation='nearest')
    plt.axis('off')
    
    # viz_sXX_lXX_idx213_main.png
    fname = f"viz_{step}_{layer}_idx{token}_{suffix}.png"
    plt.savefig(output_dir / fname, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # Hardcoded default path based on your previous run
    default_input_dir = "pico_test/qwen_results_mainpe_42000/2511-cfg-28000_cfg2_alpha0.1_baseline/24/attn_maps"
    default_output_dir = "pico_test/qwen_results_mainpe_42000/2511-cfg-28000_cfg2_alpha0.1_baseline/24/attn_viz_stand"
    
    # You confirmed index 213 for "stand"
    target_index = 213
    
    visualize_attention_maps(default_input_dir, default_output_dir, token_index=target_index)