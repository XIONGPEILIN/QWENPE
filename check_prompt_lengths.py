import json
from transformers import AutoTokenizer

# Configuration
JSON_PATH = "dataset_qwen_pe_top1000.json"
MODEL_ID = "google/siglip2-large-patch16-512"

def main():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    lengths = []
    over_64_count = 0
    max_len = 0
    longest_prompt = ""

    for item in data:
        prompt = item.get("prompt", "")
        # Tokenize without truncation to get real length
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        length = len(tokens)
        lengths.append(length)
        
        if length > 64:
            over_64_count += 1
        
        if length > max_len:
            max_len = length
            longest_prompt = prompt

    avg_len = sum(lengths) / len(lengths)
    
    print(f"Dataset Analysis for '{JSON_PATH}':")
    print(f"Total items: {len(data)}")
    print(f"Average token length: {avg_len:.2f}")
    print(f"Max token length: {max_len}")
    print(f"Items over 64 tokens: {over_64_count} ({over_64_count/len(data)*100:.2f}%)")
    
    if longest_prompt:
        print(f"\nLongest prompt (Length {max_len}):")
        print(f"{longest_prompt}")

if __name__ == "__main__":
    main()
