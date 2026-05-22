import json
import os
import requests
import base64
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
JSON_PATH = "dataset_qwen_pe_top1000.json"
OUTPUT_JSON_PATH = "dataset_qwen_pe_top1000_captioned.json"
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

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class QwenCaptionDataset(Dataset):
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

def process_item_wrapper(args):
    """Wrapper to unpack arguments for process_item."""
    index, item = args
    return index, process_item(item)

def process_item(item):
    """Worker function to process a single item."""
    import random
    
    # Simple Load Balancing
    api_url = random.choice(API_URLS)
    
    original_img_rel = item.get("image")
    # Handle edit_image being a list or string
    edit_imgs = item.get("edit_image", [])
    edited_img_rel = edit_imgs[0] if isinstance(edit_imgs, list) and edit_imgs else edit_imgs
    if isinstance(edited_img_rel, list): # Double check if nested
        edited_img_rel = edited_img_rel[0]
        
    prompt_text = item.get("prompt", "")

    if not original_img_rel or not edited_img_rel:
        return item

    original_img_path = os.path.join(BASE_IMAGE_DIR, original_img_rel)
    edited_img_path = os.path.join(BASE_IMAGE_DIR, edited_img_rel)

    if not os.path.exists(original_img_path):
        # print(f"Missing images for {original_img_rel}")
        return item

    try:
        original_b64 = encode_image(original_img_path)
    except Exception:
        return item

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for image editing description. You will output a JSON object."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_b64}"}},
                {"type": "text", "text": "This is the original image (Image 1)."},
                {
                    "type": "text", 
                    "text": f"""The editing instruction was: "{prompt_text}".

Task: Based ONLY on the original image and the instruction, predict what the edited image SHOULD look like.

Provide two CONCISE captions for evaluation (Note: The downstream encoder has a strict 64-token limit, so keep captions short and dense!):
1. 'global_caption': A short description of the entire ideal edited image.
2. 'local_caption': A short description focusing ONLY on the changed region (e.g., 'a white majestic unicorn with silver mane').

Output strictly in JSON format: {{"global_caption": "...", "local_caption": "..."}}"""
                }
            ]
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        clean_content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            json_content = json.loads(clean_content)
            item['global_caption'] = json_content.get('global_caption', '')
            item['local_caption'] = json_content.get('local_caption', '')
        except json.JSONDecodeError:
            # Fallback: try to find strict JSON bounds if there is extra text
            try:
                start = clean_content.find('{')
                end = clean_content.rfind('}')
                if start != -1 and end != -1:
                    json_content = json.loads(clean_content[start:end+1])
                    item['global_caption'] = json_content.get('global_caption', '')
                    item['local_caption'] = json_content.get('local_caption', '')
                else:
                    item['global_caption'] = content
                    item['local_caption'] = ""
            except:
                item['global_caption'] = content
                item['local_caption'] = ""
                
        return item
    except Exception as e:
        # print(f"Error processing {original_img_rel}: {e}")
        return item

def main():
    dataset = QwenCaptionDataset(JSON_PATH, BASE_IMAGE_DIR)
    
    # Load existing results for resume
    results_map = {} # Maps original image path -> processed item
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r') as f:
                existing_data = json.load(f)
                for it in existing_data:
                    # Check if it has valid captions
                    if 'global_caption' in it and it['global_caption']:
                        results_map[it['image']] = it
        except:
            print("Output file corrupt or empty, starting fresh.")
            pass

    print(f"Total items: {len(dataset)}. Already processed: {len(results_map)}")

    # Prepare list of items to process with their original indices
    # We will reconstruct the final list based on indices to preserve order
    items_to_process = []
    final_list_placeholder = [None] * len(dataset) # To store results in order
    
    for i in range(len(dataset)):
        item = dataset[i]
        img_key = item['image']
        
        if img_key in results_map:
            final_list_placeholder[i] = results_map[img_key]
        else:
            items_to_process.append((i, item))

    if not items_to_process:
        print("All items already processed.")
        return

    print(f"Items remaining to process: {len(items_to_process)}")

    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_item_wrapper, args): args[0] for args in items_to_process}
        
        save_counter = 0
        
        for future in tqdm(as_completed(futures), total=len(items_to_process), desc="Generating Captions"):
            idx, processed_item = future.result()
            final_list_placeholder[idx] = processed_item
            
            save_counter += 1
            if save_counter % 50 == 0:
                # Periodic save: filter out Nones (unprocessed) to save partial progress
                # Note: This partial save might be "holey" if we just dump final_list_placeholder
                # So we dump a clean list of what we have so far + placeholders removed?
                # Actually, better to dump the valid ones found so far plus results_map
                # But to be safe and simple: just dump everything that is not None.
                # WARNING: This changes the file length until finished.
                # Better strategy: Dump ONLY completed items for resume safety.
                current_valid = [x for x in final_list_placeholder if x is not None]
                with open(OUTPUT_JSON_PATH, 'w') as f:
                    json.dump(current_valid, f, indent=2)

    # Final Save: Ensure correct order and completeness
    final_output = [x for x in final_list_placeholder if x is not None]
    
    # Double check count
    if len(final_output) != len(dataset):
        print(f"Warning: Final count {len(final_output)} mismatches dataset count {len(dataset)}.")
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Done. Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
