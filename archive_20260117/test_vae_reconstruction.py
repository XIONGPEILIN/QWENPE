import sys
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.multiprocessing as mp
from queue import Empty
from safetensors.torch import load_file

# Add path
sys.path.append(os.path.join(os.getcwd(), "DiffSynth-Studio"))
# We need the VAE class definition
from diffsynth.models.qwen_image_vae import QwenImageVAE

def gpu_worker(gpu_id, data_indices, result_queue, json_path, base_path):
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    dtype = torch.bfloat16
    
    try:
        # Manual Loading
        vae = QwenImageVAE()
        ckpt_path = "/home/yanai-lab/xiong-p/ssd/xiong-p/models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"
        state_dict = load_file(ckpt_path)
        vae.load_state_dict(state_dict)
        vae = vae.to(device=device, dtype=dtype)
        vae.eval()
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load VAE manually: {e}")
        return

    # Load Full Data
    with open(json_path, "r") as f:
        all_data = json.load(f)
    
    # Transforms
    resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC)
    to_tensor = transforms.ToTensor()
    
    local_l1 = []
    local_l2 = []
    
    with torch.no_grad():
        for idx in tqdm(data_indices, position=gpu_id, desc=f"GPU {gpu_id}"):
            entry = all_data[idx]
            img_rel_path = entry['image']
            img_path = os.path.join(base_path, img_rel_path)
            
            if not os.path.exists(img_path):
                continue
                
            try:
                img = Image.open(img_path).convert("RGB")
                img = resize(img)
                
                img_tensor = to_tensor(img).unsqueeze(0).to(device, dtype=dtype)
                img_tensor = (img_tensor * 2.0) - 1.0
                
                # Direct VAE usage
                latents = vae.encode(img_tensor)
                recon = vae.decode(latents)
                
                diff = recon - img_tensor
                l1 = torch.abs(diff).mean().item()
                l2 = torch.pow(diff, 2).mean().item()
                
                local_l1.append(l1)
                local_l2.append(l2)
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error: {e}")
    
    result_queue.put((local_l1, local_l2))

def main():
    json_path = "dataset_qwen_pe_top1000.json"
    base_path = "pico-banana-400k-subject_driven/openimages"
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
    
    num_gpus = 8
    total_items = len(data)
    indices = np.array_split(range(total_items), num_gpus)
    
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []
    
    print(f"Starting {num_gpus} workers for {total_items} images (Manual Loading)...")
    
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=gpu_worker, 
            args=(gpu_id, indices[gpu_id], result_queue, json_path, base_path)
        )
        p.start()
        processes.append(p)
        
    all_l1 = []
    all_l2 = []
    
    finished_workers = 0
    while finished_workers < num_gpus:
        try:
            l1s, l2s = result_queue.get(timeout=60) # 1 min timeout per batch return
            all_l1.extend(l1s)
            all_l2.extend(l2s)
            finished_workers += 1
        except Empty:
            if not any(p.is_alive() for p in processes):
                break
    
    for p in processes:
        p.join()
        
    if all_l1:
        print(f"\nFinal VAE Reconstruction Metrics ({len(all_l1)} images):")
        print(f"Average L1 (MAE): {np.mean(all_l1):.6f}")
        print(f"Average L2 (MSE): {np.mean(all_l2):.6f}")
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()
