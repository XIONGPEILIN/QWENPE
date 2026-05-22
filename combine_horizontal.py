from PIL import Image
from pathlib import Path

def combine_selected_horizontal_fixed_height(target_height=1000):
    indices = [1, 6, 8, 11, 21, 23, 24]
    base_dir = Path("compare/crop_comparison_28k_vs_30k")
    save_path = base_dir / "combined_selected_horizontal_fixed_height.png"
    
    images = []
    for idx in indices:
        img_path = base_dir / f"compare_{idx}.png"
        if img_path.exists():
            img = Image.open(img_path)
            # Maintain aspect ratio while resizing to target_height
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            images.append(img_resized)
        else:
            print(f"Warning: {img_path} not found.")
            
    if not images:
        print("No images found to combine.")
        return
        
    spacing = 20
    total_width = sum(img.width for img in images) + (len(images) - 1) * spacing
    
    combined = Image.new('RGB', (total_width, target_height), (255, 255, 255))
    
    current_x = 0
    for img in images:
        combined.paste(img, (current_x, 0))
        current_x += img.width + spacing
        
    combined.save(save_path)
    print(f"Successfully saved combined image with fixed height to {save_path}")

if __name__ == "__main__":
    combine_selected_horizontal_fixed_height(target_height=800)
