import torch
import os

cache_file = "data/Qwen-Image-Edit-2511_lora-rank512-split-cache/0/0.pth"

print(f"Loading {cache_file}...")
try:
    data_tuple = torch.load(cache_file, map_location="cpu", weights_only=False)
    print(f"Data type: {type(data_tuple)}")
    if isinstance(data_tuple, tuple):
        print(f"Tuple len: {len(data_tuple)}")
        data = data_tuple[0] # inputs_shared
    else:
        data = data_tuple

    print("Keys:", data.keys())
    
    if "edit_latents" in data:
        el = data["edit_latents"]
        if isinstance(el, list):
            print(f"edit_latents is LIST. Len: {len(el)}")
            if len(el) > 0:
                print(f"edit_latents[0] Type: {type(el[0])}")
                if hasattr(el[0], 'shape'):
                    print(f"edit_latents[0] Shape: {el[0].shape}")
        else:
            print(f"edit_latents Type: {type(el)}")
            if hasattr(el, 'shape'):
                print(f"edit_latents Shape: {el.shape}")
            
    if "processed_inpaint_mask" in data:
        mask = data["processed_inpaint_mask"]
        print(f"processed_inpaint_mask Type: {type(mask)}")
        if hasattr(mask, 'shape'):
            print(f"processed_inpaint_mask Shape: {mask.shape}")
        
except Exception as e:
    print(f"Error: {e}")