import pandas as pd

data = {
    "Task": ["Inpainting", "Local Subject Addition", "Local Subject Removal", "Local Text Removal", "Local Text Render"],
    "aes_v2.5": [4.353530, 5.114608, 5.021658, 4.800064, 4.330330],
    "musiq_koniq": [61.674982, 62.391083, 58.158275, 58.061950, 44.016858],
    "clip_cap": [0.267331, 0.276779, 0.260421, 0.270548, 0.281760],
    "vllmqa": [0.765586, 0.641148, 0.633663, 0.632653, 0.490798],
    "clip_src": [0.838662, 0.887129, 0.873895, 0.949094, 0.901521],
    "l1_raw": [0.016663, 0.016649, 0.015996, 0.016509, 0.010693]
}

df = pd.DataFrame(data)

averages = df.mean(numeric_only=True)

print("### Average Metrics across 5 Tasks")
print(f"- Aesthetic↑: {averages['aes_v2.5']:.6f}")
print(f"- Imaging↑: {averages['musiq_koniq']:.6f}")
print(f"- CLIP-cap↑: {averages['clip_cap']:.6f}")
print(f"- VLLM-QA↑: {averages['vllmqa']:.6f}")
print(f"- CLIP-src↑: {averages['clip_src']:.6f}")
print(f"- L1-src↓: {averages['l1_raw']:.6f}")
