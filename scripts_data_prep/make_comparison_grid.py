import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def make_grid():
    root_dir = Path("test-cfg4")
    steps = ["8000", "14000", "15000"]
    output_dir = Path("comparison_grid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有步骤文件夹中共有的样本 ID
    sample_sets = []
    for step in steps:
        step_path = root_dir / step
        if not step_path.exists():
            print(f"警告: 目录 {step_path} 不存在。")
            return
        samples = {d.name for d in step_path.iterdir() if d.is_dir() and d.name.isdigit()}
        sample_sets.append(samples)
    
    common_samples = set.intersection(*sample_sets)
    sorted_samples = sorted(list(common_samples), key=int)
    
    print(f"找到 {len(sorted_samples)} 个共同样本: {sorted_samples}")

    for sample_id in sorted_samples:
        images = []
        valid = True
        for step in steps:
            img_path = root_dir / step / sample_id / "output.png"
            if not img_path.exists():
                print(f"样本 {sample_id} 在步骤 {step} 中缺少图片")
                valid = False
                break
            
            img = Image.open(img_path).convert("RGB")
            
            # 在图片左上角添加步骤标记
            draw = ImageDraw.Draw(img)
            try:
                # 尝试加载默认字体
                font = ImageFont.load_default()
                draw.text((10, 10), f"Step {step}", fill=(255, 0, 0), font=font)
            except Exception:
                pass
            
            images.append(img)
        
        if not valid:
            continue

        # 确保所有图片大小一致
        base_w, base_h = images[0].size
        resized_images = [img.resize((base_w, base_h)) for img in images]

        # 创建水平拼接图 (1 行 3 列)
        total_width = base_w * len(steps)
        combined_img = Image.new("RGB", (total_width, base_h))

        for idx, img in enumerate(resized_images):
            combined_img.paste(img, (idx * base_w, 0))

        save_path = output_dir / f"sample_{sample_id}.png"
        combined_img.save(save_path)
        print(f"已保存样本 {sample_id} 的对比图至 {save_path}")

if __name__ == "__main__":
    make_grid()
