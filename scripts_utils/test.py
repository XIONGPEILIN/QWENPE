import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# 1. 配置与路径工具 (复用你的代码)
# ---------------------------------------------------------
REPO_ROOT_SENTINEL = "dataset_qwen_pe.json"

def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(10):
        if (cur / REPO_ROOT_SENTINEL).exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(f"Could not locate {REPO_ROOT_SENTINEL} upwards from {start}")

def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("RGB")

def tensor_to_pil(tensor):
    """将 Tensor (-1~1) 转回 PIL"""
    image = tensor.cpu().permute(0, 2, 3, 1).float()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image[0])

# ---------------------------------------------------------
# 2. 主测试逻辑
# ---------------------------------------------------------
def main():
    repo_root = find_repo_root(Path(__file__).resolve())
    
    # 模拟你的环境路径加载
    sys.path.append(str(repo_root / "DiffSynth-Studio"))
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    # --- 用户配置区域 ---
    # 假设 image_folder 是空或者特定的子目录，根据你原本代码上下文调整
    # 如果你的 dataset json 在根目录，且图片路径是相对根目录的，留空即可
    image_folder = "pico-banana-400k-subject_driven/openimages"
    
    sample = {
        "image": "edited/sft/62651.png", 
        # 其他字段测试 VAE 时不需要
    }
    # -------------------

    # 1. 加载原图
    image_path = repo_root / image_folder / sample["image"]
    print(f">>> 正在读取原图: {image_path}")
    original_pil = load_rgb(image_path)
    print(f"    原图尺寸: {original_pil.size}")

    # 2. 初始化 Pipeline (只加载 VAE 相关的即可，但为了兼容性保持原样)
    print(">>> 正在加载模型 (VAE)...")
    # 注意：这里假设 QwenImagePipeline 支持从 model_manager 或直接路径加载
    # 为了简化，这里直接初始化 pipeline，确保 vae 被加载
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    device = pipe.device
    dtype = pipe.torch_dtype

    # 创建输出目录
    output_dir = repo_root / "vae_real_image_test"
    output_dir.mkdir(exist_ok=True)
    
    # 保存原始大图作为参考
    original_pil.save(output_dir / "original_reference.png")

    # ---------------------------------------------------------
    # 3. 循环测试分辨率 (16 -> 128)
    # ---------------------------------------------------------
    resolutions = range(16, 256 + 1, 16) # [16, 32, 48, 64, 80, 96, 112, 128]

    print(f"\n>>> 开始测试分辨率: {list(resolutions)}")
    
    for res in resolutions:
        try:
            # A. 强制 Resize 原图到测试分辨率 (模拟极小分辨率输入)
            # 使用 LANCZOS 确保缩小时质量最好，排除 resize 算法造成的模糊
            input_pil = original_pil.resize((res, res), Image.LANCZOS)
            
            # 保存一下“输入给 VAE 的图”，方便对比 VAE 到底恶化了多少
            input_pil.save(output_dir / f"res_{res}_input.png")
            image = pipe.preprocess_image(input_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
            # B. 预处理 -> Tensor

            # C. VAE Encode -> Decode
            with torch.no_grad():
                latents = pipe.vae.encode(image)
                
                # 解码
                decoded_output = pipe.vae.decode(latents)

            # D. 保存结果
            recon_img = tensor_to_pil(decoded_output)
            save_path = output_dir / f"res_{res}_vae_output.png"
            recon_img.save(save_path)
            
            latent_size = latents.shape[-1]
            print(f"[{res}x{res}] -> Latent[{latent_size}x{latent_size}] -> 保存至: {save_path.name}")

        except Exception as e:
            print(f"[{res}x{res}] 失败: {e}")

    print(f"\n>>> 测试完成。请查看目录: {output_dir}")

if __name__ == "__main__":
    main()