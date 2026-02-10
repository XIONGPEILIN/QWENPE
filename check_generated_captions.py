import json
from transformers import AutoTokenizer
import os

# Configuration
JSON_PATH = "dataset_qwen_pe_top1000_captioned.json"
MODEL_ID = "google/siglip2-large-patch16-512"

def analyze_captions(caption_key, data, tokenizer):
    lengths = []
    over_64 = 0
    max_len = 0
    longest = ""
    empty_count = 0

    for item in data:
        cap = item.get(caption_key, "")
        if not cap:
            empty_count += 1
            continue
            
        tokens = tokenizer.encode(cap, add_special_tokens=True)
        length = len(tokens)
        lengths.append(length)
        
        if length > 64:
            over_64 += 1
        
        if length > max_len:
            max_len = length
            longest = cap

    if not lengths:
        print(f"--- {caption_key} Analysis: No data ---")
        return

    avg_len = sum(lengths) / len(lengths)
    
    print(f"\n--- {caption_key} Analysis ---")
    print(f"Valid items: {len(lengths)} (Empty: {empty_count})")
    print(f"Average token length: {avg_len:.2f}")
    print(f"Max token length: {max_len}")
    print(f"Items over 64 tokens: {over_64} ({over_64/len(lengths)*100:.2f}%)")
    
    if longest:
        print(f"Longest caption ({max_len} tokens): \"{longest}\"")

def main():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    if not os.path.exists(JSON_PATH):
        print(f"File not found: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} generated items from '{JSON_PATH}'...")
    
    analyze_captions("global_caption", data, tokenizer)
    analyze_captions("local_caption", data, tokenizer)

if __name__ == "__main__":
    main()
