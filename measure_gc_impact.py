import torch
from diffsynth.models.qwen_image_dit import QwenImageDiT, STE
from diffsynth.core.gradient.gradient_checkpoint import gradient_checkpoint_forward
import sys

def measure_layer_mem():
    device = "cuda:1"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    
    # 1. Load Model (Only 1 layer needed for measurement to save time/mem)
    print("Loading 1-layer model...")
    model = QwenImageDiT(num_layers=1).to(device=device, dtype=dtype)
    block = model.transformer_blocks[0]
    
    # 2. Prepare Inputs (L=8704)
    B = 1
    Seq = 8704
    Dim = 3072
    image = torch.randn(B, Seq, Dim, device=device, dtype=dtype).requires_grad_(True)
    text = torch.randn(B, 512, Dim, device=device, dtype=dtype).requires_grad_(True)
    temb = torch.randn(B, Dim, device=device, dtype=dtype).requires_grad_(True)
    
    # Dummy rotary embeddings
    freqs = torch.randn(Seq, 64, device=device, dtype=torch.float32)
    rotary = (freqs, freqs[:512])

    print("\n--- Measurement Start ---")
    
    # --- Measure No GC ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_0 = torch.cuda.memory_allocated()
    
    # Forward pass (Standard)
    with torch.cuda.amp.autocast(dtype=dtype):
        out_text, out_image = block(image, text, temb, image_rotary_emb=rotary)
        
    mem_1 = torch.cuda.memory_allocated()
    no_gc_usage = (mem_1 - mem_0) / 1024**2
    print(f"Layer Activation (No GC): {no_gc_usage:.2f} MB")
    
    # Cleanup to ensure clean state
    del out_text, out_image
    torch.cuda.empty_cache()
    
    # --- Measure With GC ---
    torch.cuda.reset_peak_memory_stats()
    mem_2 = torch.cuda.memory_allocated()
    
    with torch.cuda.amp.autocast(dtype=dtype):
        # use_gradient_checkpointing=True
        # We simulate the exact call
        out_text_gc, out_image_gc = gradient_checkpoint_forward(
            block, 
            True, # use_gc
            False, # offload
            image=image, text=text, temb=temb, image_rotary_emb=rotary
        )
        
    mem_3 = torch.cuda.memory_allocated()
    with_gc_usage = (mem_3 - mem_2) / 1024**2
    print(f"Layer Activation (With GC): {with_gc_usage:.2f} MB")
    
    return no_gc_usage, with_gc_usage

if __name__ == "__main__":
    try:
        no_gc, with_gc = measure_layer_mem()
        
        # Calculation
        total_vram = 96 * 1024 # MB
        # Fixed costs: 
        # Model Weights (bf16) = 7B * 2 = 14GB (ZeRO-2 keeps full model weights? Yes)
        # Optimizer States (ZeRO-2 split on 5 GPUs) = (7B*12)/5 = 16.8GB
        # Gradients (ZeRO-2 split) = (7B*2)/5 = 2.8GB
        # Total Static = 14 + 16.8 + 2.8 = 33.6 GB
        
        static_total = 33.6 * 1024 
        available = total_vram - static_total
        
        print(f"\n--- Analysis (96GB VRAM) ---")
        print(f"Static Overhead (Weights+Opt+Grad): {static_total/1024:.2f} GB")
        print(f"Available for Activations: {available/1024:.2f} GB")
        print(f"Cost per Layer (No GC): {no_gc:.2f} MB")
        print(f"Cost per Layer (With GC): {with_gc:.2f} MB")
        
        # Solve for x (No GC layers):
        # x * no_gc + (60 - x) * with_gc <= available
        # x * (no_gc - with_gc) <= available - 60 * with_gc
        
        max_no_gc_layers = (available - 60 * with_gc) / (no_gc - with_gc)
        print(f"\nTheoretical Max 'No GC' Layers: {int(max_no_gc_layers)}")
        
        # Safety Margin Analysis
        # If we set safety margin to 20GB, what happens?
        # Stop 'No GC' when available < 20GB.
        # Initial available = 62.4 GB.
        # Run until 20GB left -> used 42.4 GB.
        # 42.4 GB / no_gc = layers
        
        used_capacity = (available/1024) - 20
        layers_dynamic = (used_capacity * 1024) / no_gc
        print(f"With 20GB Margin, Dynamic Strategy will run approx {int(layers_dynamic)} layers without GC.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
