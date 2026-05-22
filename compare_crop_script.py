from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

def get_qwen_bbox(back_mask, width, height):
    back_mask = back_mask.resize((width, height), resample=Image.NEAREST)
    back_mask = back_mask.convert("L")
    bbox = back_mask.getbbox()
    
    if bbox is None:
        left, upper, right, lower = 0, 0, width, height
    else:
        left, upper, right, lower = bbox
    
    new_left = (left // 16) * 16
    new_upper = (upper // 16) * 16
    new_right = ((right + 16 - 1) // 16) * 16
    new_lower = ((lower + 16 - 1) // 16) * 16
    
    new_left = max(0, new_left)
    new_upper = max(0, new_upper)
    new_right = min(width, new_right)
    new_lower = min(height, new_lower)
    
    return (new_left, new_upper, new_right, new_lower)

def process_all():
    base_28 = Path("compare/grid_search_ablation/2011-ste-28000_cfg2_alpha0.1_baseline")
    base_30 = Path("compare/new-woste-30000_cfg2")
    save_dir = Path("compare/crop_comparison_28k_vs_30k")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有数字命名的子目录
    subdirs = sorted([d.name for d in base_28.iterdir() if d.is_dir() and d.name.isdigit()], key=int)
    
    for subdir in subdirs:
        dir_28 = base_28 / subdir
        dir_30 = base_30 / subdir
        
        if not dir_30.exists():
            continue
            
        print(f"Processing directory: {subdir}")
        
        try:
            mask = Image.open(dir_28 / "back_mask.png")
            out_28 = Image.open(dir_28 / "output.png")
            sub_28 = Image.open(dir_28 / "output_sub.png")
            
            out_30 = Image.open(dir_30 / "output.png")
            sub_30 = Image.open(dir_30 / "output_sub.png")
            
            width, height = out_28.size
            bbox = get_qwen_bbox(mask, width, height)
            
            # Crop output images
            crop_28 = out_28.crop(bbox)
            crop_30 = out_30.crop(bbox)
            
            # Ensure sub images match the crop size (they should, but just in case)
            # Or at least have the same width for vertical stacking
            target_w = crop_28.width
            target_h = crop_28.height
            
            # 创建左侧列 (28k)
            col_28 = Image.new('RGB', (target_w, target_h * 2 + 10), (255, 255, 255))
            col_28.paste(crop_28, (0, 0))
            col_28.paste(sub_28.resize((target_w, target_h)), (0, target_h + 10))
            
            # 创建右侧列 (30k)
            col_30 = Image.new('RGB', (target_w, target_h * 2 + 10), (255, 255, 255))
            col_30.paste(crop_30, (0, 0))
            col_30.paste(sub_30.resize((target_w, target_h)), (0, target_h + 10))
            
            # 合并左右
            combined = Image.new('RGB', (target_w * 2 + 10, target_h * 2 + 10), (255, 255, 255))
            combined.paste(col_28, (0, 0))
            combined.paste(col_30, (target_w + 10, 0))
                
            combined.save(save_dir / f"compare_{subdir}.png")
            
        except Exception as e:
            print(f"Error processing {subdir}: {e}")

if __name__ == "__main__":
    process_all()
