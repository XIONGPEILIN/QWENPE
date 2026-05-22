"""
小规模测试脚本：从数据集中随机抽 10 条，运行 reverse prompt 流程，打印结果。
用法: python test_reverse_small.py
"""
import json
import os
import random
import sys

# 复用 batch_reverse_prompts 中的核心函数
from batch_reverse_prompts import (
    process_item, BASE_IMAGE_DIR, strip_picture1_prefix
)

JSON_PATH = "dataset_qwen_pe_train_crop.json"
N_SAMPLES = 10

def main():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # 随机选 N 条
    samples = random.sample(data, min(N_SAMPLES, len(data)))
    
    print(f"=== 测试 {len(samples)} 条随机样本 ===\n")
    
    success = 0
    fail = 0
    
    for i, item in enumerate(samples):
        print(f"--- [{i+1}/{len(samples)}] ---")
        print(f"  原始 prompt: {item.get('prompt', '')[:100]}...")
        print(f"  image: {item.get('image', '')}")
        print(f"  edit_image: {item.get('edit_image', '')}")
        
        result = process_item(item)
        
        if result is None:
            print(f"  ❌ 失败 (返回 None)")
            fail += 1
            continue
        
        prompt = result.get('prompt', '')
        gc = result.get('global_caption', '')
        lc = result.get('local_caption', '')
        
        # 检查质量：只检测真正的思考文本关键词
        has_thinking = any(kw in prompt for kw in ['The user wants me to', '**Input:**', 'Let me ', 'Wait,', 'Let\'s'])
        
        status = "✅" if not has_thinking and gc and lc else "⚠️"
        if has_thinking:
            status = "❌ (思考文本泄露!)"
        
        gc_words = len(gc.split())
        lc_words = len(lc.split())
        
        print(f"  {status}")
        print(f"  reversed prompt: {prompt}")
        print(f"  global_caption ({gc_words}w): {gc}")
        print(f"  local_caption  ({lc_words}w): {lc}")
        print()
        
        if not has_thinking and gc and lc:
            success += 1
        else:
            fail += 1
    
    print(f"\n=== 结果: {success}/{len(samples)} 成功, {fail}/{len(samples)} 失败/部分缺失 ===")

if __name__ == "__main__":
    main()
