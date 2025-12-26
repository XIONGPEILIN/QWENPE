from PIL import Image, ImageChops
import numpy as np

def test_mask_logic(image_path, output_path):
    print(f"Testing logic on: {image_path}")
    input_mask = Image.open(image_path)
    
    # 模拟代码中的 inter_width, inter_height (使用原图尺寸)
    inter_width, inter_height = input_mask.size
    print(f"Size: {inter_width}x{inter_height}")

    # ---------------------------------------------------------
    # 提取自 app_gradio_multi.py / app_flux_fill.py 的核心逻辑
    # ---------------------------------------------------------
    if input_mask.mode in ('RGBA', 'LA') or (input_mask.mode == 'P' and 'transparency' in input_mask.info):
        alpha = input_mask.convert('RGBA').split()[-1]
        
        # 检查 Alpha 通道极值
        extrema = alpha.getextrema()
        print(f"Alpha Extrema: {extrema}")

        if extrema != (255, 255):
            print("Logic: Alpha channel has transparency.")
            print("Applying Inverted Logic: Alpha < 255 -> Mask=255 (Edit), Alpha=255 -> Mask=0 (Keep)")
            
            # 核心反转逻辑
            raw_mask = alpha.point(lambda p: 255 if p < 255 else 0)
            
            # Resize (这里实际上尺寸没变，但也模拟一下流程)
            raw_mask = raw_mask.resize((inter_width, inter_height), Image.NEAREST)
        else:
            print("Logic: Alpha channel is fully opaque.")
            print("Applying Traditional Logic: Grayscale conversion.")
            raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)
    else:
        print("Logic: No Alpha channel.")
        print("Applying Traditional Logic: Grayscale conversion.")
        raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)
    # ---------------------------------------------------------

    # 保存结果
    raw_mask.save(output_path)
    print(f"Result saved to: {output_path}")
    
    # 简单统计 Mask 结果
    mask_arr = np.array(raw_mask)
    white_pixels = np.sum(mask_arr == 255)
    total_pixels = mask_arr.size
    print(f"Mask White Pixels (Edit Area): {white_pixels} ({white_pixels/total_pixels:.2%})")
    print(f"Mask Black Pixels (Keep Area): {total_pixels - white_pixels}")

if __name__ == "__main__":
    test_mask_logic("30-mask1.png", "extracted_test_mask.png")
