import torch
import sys, os

# Add path
sys.path.append(os.path.join(os.getcwd(), "DiffSynth-Studio"))
from diffsynth.pipelines.qwen_image import model_fn_qwen_image
from diffsynth.models.qwen_image_dit import QwenImageDiT, STE

def test_gc_capacity_and_prediction():
    device = "cuda"
    dtype = torch.bfloat16
    
    print("--- 1. Initializing Models (8GB LoRA Simulation) ---")
    # Load Real DiT architecture
    dit = QwenImageDiT(num_layers=60).to(device=device, dtype=dtype)
    ste = STE().to(device=device, dtype=dtype)
    dit.train(); ste.train()

    # Simulation setup: 1024x1024 dual stream
    height, width = 1024, 1024
    batch_size = 1
    latents = torch.randn(batch_size, 16, 128, 128, device=device, dtype=dtype)
    sub_latents = torch.randn(batch_size, 16, 128, 128, device=device, dtype=dtype)
    subyx = (0, 64, 0, 64) 
    prompt_emb = torch.randn(batch_size, 512, 3584, device=device, dtype=dtype)
    prompt_emb_mask = torch.ones(batch_size, 512, device=device, dtype=torch.long)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)

    print(f"--- 2. VRAM Status Analysis ---")
    torch.cuda.empty_cache()
    base_rsrv = torch.cuda.memory_reserved(device) / (1024**3)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    print(f"Baseline Reserved (Model Weights): {base_rsrv:.2f} GB")
    print(f"Total GPU Memory: {total_mem:.2f} GB")
    
    # Simulate Step 3 Steady State
    dit._log_step = 3 

    print(f"\n--- 3. Execution Test (Watch Layer Logs) ---")
    try:
        with torch.no_grad():
            model_fn_qwen_image(
                dit=dit,
                ste=ste,
                latents=latents,
                sub_latents=sub_latents,
                subyx=subyx,
                timestep=timestep,
                prompt_emb=prompt_emb,
                prompt_emb_mask=prompt_emb_mask,
                height=height,
                width=width,
                use_gradient_checkpointing=True,
                zero_cond_t=True
            )
    except RuntimeError as e:
        print(f"Test Interrupted: {e}")

    # 4. Math Calculation
    # Token count: (1024/16)^2 * 2 + 512 = 8704
    # Est Act: 8704 * 3072 * 32 bytes = 0.8 GB/layer
    act_per_layer = (8704 * 3072 * 32) / (1024**3)
    buffer = 7.5
    
    # Available space
    available = total_mem - base_rsrv - buffer
    layers_can_off = max(0, int(available / act_per_layer))
    
    print(f"\n--- 4. Prediction Report ---")
    print(f"Activation per Layer: ~{act_per_layer:.3f} GB")
    print(f"Available Space for Acts (after 7.5G buffer): {available:.2f} GB")
    print(f"Predicted GC OFF Layers: {layers_can_off}")
    if layers_can_off < 60:
        print(f"Recommendation: KEEP GC OFF for the FIRST {layers_can_off} layers, then ON for the rest.")
    else:
        print(f"Recommendation: Plenty of space. You can try disabling GC entirely.")

if __name__ == "__main__":
    test_gc_capacity_and_prediction()