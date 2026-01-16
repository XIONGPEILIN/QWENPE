import torch
import torch.nn as nn
from diffsynth.models.qwen_image_dit import QwenImageDiT, STE
import os

def print_memory_stats(label):
    allocated = torch.cuda.memory_allocated(1) / 1024**2
    reserved = torch.cuda.memory_reserved(1) / 1024**2
    print(f"[{label}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    return allocated

def measure():
    device = "cuda:1"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    
    print("--- 初始状态 ---")
    base_mem = print_memory_stats("Initial")
    
    print("\n--- 加载模型 (DiT + STE) ---")
    # Qwen-Image-Edit-2511 默认 60 层
    model = QwenImageDiT(num_layers=60).to(device=device, dtype=dtype)
    ste = STE(num_layers=60).to(device=device, dtype=dtype)
    
    model_mem = print_memory_stats("After Loading Model")
    total_params_mem = model_mem - base_mem
    print(f"模型参数总占用: {total_params_mem:.2f} MB (~{total_params_mem/1024:.2f} GB)")
    print(f"平均每层参数占用: {total_params_mem/60:.2f} MB")

    # 模拟输入 (LBM 训练: 1024x1024 主图 + 50% 面积 Sub Crop)
    # Main Stream: 4096 tokens
    # Sub Stream:  2048 tokens
    # Total Image Tokens: 6144
    
    batch_size = 1
    seq_len = 6144 
    text_len = 512 # Text tokens
    hidden_dim = 3072
    
    image = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    text = torch.randn(batch_size, text_len, hidden_dim, device=device, dtype=dtype) 
    temb = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    
    print(f"\n--- 模拟前向传播 (估算一层激活值, L={seq_len}) ---")
    torch.cuda.empty_cache()
    before_forward = torch.cuda.memory_allocated(1)
    
    # 只跑一层 block
    block = model.transformer_blocks[0]
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        # 模拟 forward 过程
        out_text, out_image = block(image, text, temb)
        
    after_forward = torch.cuda.memory_allocated(1)
    act_mem = (after_forward - before_forward) / 1024**2
    print(f"单层 Block 激活值占用 (BS=1): {act_mem:.2f} MB")
    
    total_est_act = act_mem * 60
    print(f"全量 60 层总激活值估算 (无 GC): {total_est_act:.2f} MB (~{total_est_act/1024:.2f} GB)")
    
    print("\n--- 结论分析 (96GB VRAM) ---")
    print(f"静态权重: {total_params_mem/1024:.2f} GB")
    print(f"激活值 (全量): {total_est_act/1024:.2f} GB")
    print(f"总计 (不带梯度): {(total_params_mem + total_est_act)/1024:.2f} GB")
    print("注意：反向传播时，显存占用会翻倍（存储梯度 + 额外的中间变量）。")

if __name__ == "__main__":
    measure()
