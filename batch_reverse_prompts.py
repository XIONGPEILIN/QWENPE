import json
import os
import re
import requests
import base64
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
JSON_PATH = "dataset_qwen_pe_train_crop.json"
OUTPUT_JSON_PATH = "dataset_qwen_pe_reversed.json"
BASE_IMAGE_DIR = "/host/ssd2/xiong-p/qwenpe/pico-banana-400k-subject_driven/openimages"
# List of API URLs for load balancing
API_URLS = [
    "http://localhost:7512/v1/chat/completions",
    "http://localhost:7513/v1/chat/completions",
    "http://localhost:7514/v1/chat/completions",
    "http://localhost:7515/v1/chat/completions"
]
MODEL_NAME = "Qwen/Qwen3.5-27B"
NUM_THREADS = 16  # High concurrency

# ── Helper Functions ──────────────────────────────────────────────────────────

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def strip_thinking(text):
    """Strip thinking/reasoning content from model output."""
    # Remove <think>...</think> blocks (if any)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def extract_json_value(text, key):
    """Extract a value for a specific key from text that may contain JSON."""
    # Try standard JSON parse first
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            obj = json.loads(text[start:end+1])
            val = obj.get(key, '')
            if val:
                return val
    except:
        pass
    # Fallback: regex extraction for "key": "value"
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    m = re.search(pattern, text)
    if m:
        return m.group(1)
    return ''

def clean_model_response(raw_content):
    """Clean raw model response: strip thinking, remove markdown fences."""
    content = strip_thinking(raw_content)
    content = content.replace("```json", "").replace("```", "").strip()
    return content

def strip_picture1_prefix(prompt_text):
    """Remove 'Picture 1 is the image to modify. ' prefix if present."""
    prefix = "Picture 1 is the image to modify. "
    if prompt_text.startswith(prefix):
        return prompt_text[len(prefix):].strip()
    prefix2 = "Picture 1 is the image to modify."
    if prompt_text.startswith(prefix2):
        return prompt_text[len(prefix2):].strip()
    return prompt_text.strip()

# ── Dataset ───────────────────────────────────────────────────────────────────

class QwenDataset(Dataset):
    def __init__(self, json_path, base_dir):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Input JSON not found: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.base_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item.copy()

# ── Processing ────────────────────────────────────────────────────────────────

def process_item_wrapper(args):
    """Wrapper to unpack arguments for process_item."""
    index, item = args
    return index, process_item(item)

def process_item(item):
    """Worker function to process a single item for reverse prompting."""
    import random
    
    api_url = random.choice(API_URLS)
    
    # Extract original paths
    # 原本的 `image` 是生成的 GT图, `edit_image` 是真实的输入图
    # 在逆向任务中: 新的输入是生成的图(old image), 新的目标是真实的图(old edit_image)
    old_output_img_rel = item.get("image") 
    
    old_input_imgs = item.get("edit_image", [])
    old_input_img_rel = old_input_imgs[0] if isinstance(old_input_imgs, list) and old_input_imgs else old_input_imgs
    if isinstance(old_input_img_rel, list):
        old_input_img_rel = old_input_img_rel[0]
        
    original_prompt_text = item.get("prompt", "")
    # Strip "Picture 1 is the image to modify." prefix for cleaner prompt to model
    clean_original_prompt = strip_picture1_prefix(original_prompt_text)

    if not old_output_img_rel or not old_input_img_rel:
        return None

    # Resolve absolute paths
    new_input_img_path = os.path.join(BASE_IMAGE_DIR, old_output_img_rel)
    new_target_img_path = os.path.join(BASE_IMAGE_DIR, old_input_img_rel)

    if not os.path.exists(new_input_img_path) or not os.path.exists(new_target_img_path):
        return None

    try:
        new_input_b64 = encode_image(new_input_img_path)
        new_target_b64 = encode_image(new_target_img_path)
    except Exception:
        return None

    # ── Phase 1: Reverse Prompt ───────────────────────────────────────────
    messages_1 = [
        {
            "role": "system",
            "content": "You are a professional image editing assistant. Your task is to analyze two images and reverse an editing instruction. You must output strictly a JSON object."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new_input_b64}"}},
                {"type": "text", "text": "Image 1: Starting Image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new_target_b64}"}},
                {"type": "text", "text": "Image 2: Target Ground Truth Image."},
                {
                    "type": "text", 
                    "text": f"""The original editing instruction that turned Image 2 into Image 1 was: "{clean_original_prompt}".

Your task is to write the REVERSE instruction: how to edit Image 1 back into Image 2. Carefully observe Image 2. If an object needs to be added or changed, you MUST describe its specific visual details (color, texture, shape) as it appears in Image 2. If something needs to be removed, state clearly what it is. Keep the instruction concise and directly actionable.

Return STRICTLY a JSON object in this format:
{{"reversed_prompt": "..."}}"""
                }
            ]
        }
    ]

    payload_1 = {
        "model": MODEL_NAME,
        "messages": messages_1,
        "max_tokens": 512,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    try:
        response_1 = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload_1), timeout=90)
        response_1.raise_for_status()
        raw_content_1 = response_1.json()['choices'][0]['message']['content']
        clean_content_1 = clean_model_response(raw_content_1)
        
        # Extract reversed_prompt using robust extraction
        reversed_prompt = extract_json_value(clean_content_1, 'reversed_prompt')
        
        if not reversed_prompt:
            # Fallback: use the raw text as the reversed prompt
            reversed_prompt = clean_content_1
            if not reversed_prompt:
                return None
                
        # ── Phase 2: Global Caption (clean context, only target image) ─────────
        messages_2 = [
            {
                "role": "system",
                "content": "You are a helpful assistant for image editing description. You will output a JSON object."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new_target_b64}"}},
                    {"type": "text", "text": "This is the target image."},
                    {
                        "type": "text", 
                        "text": f"""The editing instruction applied to create this image was: "{reversed_prompt}".

Provide a single SHORT caption describing the entire image. STRICT RULE: The caption MUST be under 20 words. Example: 'A sunny beach with a red umbrella and two lounge chairs on white sand.'

Output strictly in JSON format: {{"global_caption": "..."}}"""
                    }
                ]
            }
        ]
        
        payload_2 = {
            "model": MODEL_NAME,
            "messages": messages_2,
            "max_tokens": 128,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        
        response_2 = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload_2), timeout=90)
        response_2.raise_for_status()
        raw_content_2 = response_2.json()['choices'][0]['message']['content']
        clean_content_2 = clean_model_response(raw_content_2)
        
        global_caption = extract_json_value(clean_content_2, 'global_caption')

        # ── Phase 3: Local Caption (clean context, only target image) ─────────
        messages_3 = [
            {
                "role": "system",
                "content": "You are a helpful assistant for image editing description. You will output a JSON object."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new_target_b64}"}},
                    {"type": "text", "text": "This is the target image."},
                    {
                        "type": "text", 
                        "text": f"""The editing instruction applied to create this image was: "{reversed_prompt}".

Provide a single SHORT caption focusing ONLY on the changed region. STRICT RULE: The caption MUST be under 15 words. Example: 'a white majestic unicorn with silver mane'

Output strictly in JSON format: {{"local_caption": "..."}}"""
                    }
                ]
            }
        ]
        
        payload_3 = {
            "model": MODEL_NAME,
            "messages": messages_3,
            "max_tokens": 128,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        
        response_3 = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload_3), timeout=90)
        response_3.raise_for_status()
        raw_content_3 = response_3.json()['choices'][0]['message']['content']
        clean_content_3 = clean_model_response(raw_content_3)
        
        local_caption = extract_json_value(clean_content_3, 'local_caption')
                
        # 构建对调后的最终结构
        new_item = {
            "prompt": f"Picture 1 is the image to modify. {reversed_prompt}",
            "original_prompt": original_prompt_text,
            "image": old_input_img_rel,             # 新的 GT 是真实的图像
            "edit_image": [old_output_img_rel],     # 新的 输入 是原本生成的图像
            "ref_gt": item.get("ref_gt", ""),       # 保留其他 metadata
            "back_mask": item.get("back_mask", ""),
            "global_caption": global_caption,
            "local_caption": local_caption
        }
        return new_item
        
    except Exception as e:
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dataset = QwenDataset(JSON_PATH, BASE_IMAGE_DIR)
    
    # Load existing results for resume
    results_map = {} 
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r') as f:
                existing_data = json.load(f)
                for it in existing_data:
                    if 'original_prompt' in it and it['original_prompt']:
                        # 唯一键使用前输入后输出组合
                        key = f"{it['edit_image'][0]}_{it['image']}"
                        # Only accept items that are actually valid (no thinking text)
                        prompt = it.get('prompt', '')
                        if len(prompt) <= 300 and 'The user wants me to' not in prompt:
                            results_map[key] = it
        except:
            print("Output file corrupt or empty, starting fresh.")
            pass

    print(f"Total items: {len(dataset)}. Already processed (valid): {len(results_map)}")

    items_to_process = []
    final_list_placeholder = [None] * len(dataset)
    
    for i in range(len(dataset)):
        item = dataset[i]
        old_output = item.get("image")
        old_input = item.get("edit_image", [""])[0]
        if isinstance(old_input, list): old_input = old_input[0]
        
        key = f"{old_output}_{old_input}"
        
        if key in results_map:
            final_list_placeholder[i] = results_map[key]
        else:
            items_to_process.append((i, item))

    if not items_to_process:
        print("All items already processed.")
        return

    print(f"Items remaining to process: {len(items_to_process)}")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_item_wrapper, args): args[0] for args in items_to_process}
        
        save_counter = 0
        for future in tqdm(as_completed(futures), total=len(items_to_process), desc="Reversing Prompts"):
            idx, processed_item = future.result()
            
            final_list_placeholder[idx] = processed_item
            
            save_counter += 1
            if save_counter % 50 == 0:
                current_valid = [x for x in final_list_placeholder if x is not None]
                with open(OUTPUT_JSON_PATH, 'w') as f:
                    json.dump(current_valid, f, indent=2)

    # Final Save
    final_output = [x for x in final_list_placeholder if x is not None]
    
    if len(final_output) != len(dataset):
        print(f"Warning: Final count {len(final_output)} mismatches dataset count {len(dataset)}. Some items failed or were missing images.")
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Done. Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
