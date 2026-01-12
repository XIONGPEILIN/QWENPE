import os
import json
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import cpu_count

# 目标目录
TARGET_DIR = "pico-banana-400k-subject_driven/openimages/dino_mask_audit_1"
ABS_TARGET_DIR = "/home/yanai-lab/xiong-p/ssd/xiong-p/qwenpe/pico-banana-400k-subject_driven/openimages/dino_mask_audit_1"

def check_file(filepath):
    """
    检查单个文件是否满足条件。
    返回 filepath 如果满足，否则返回 None。
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return None

    results = data.get("results", {})
    if not results:
        return None

    global_res = results.get("global", {})
    
    # 条件 1: background_changed 必须为 True
    if global_res.get("background_changed") is not False:
        return None
    
    # 条件 2: remove 和 add 中，sub_mask_results 数量 > 1 且包含 object_unchanged == True
    def is_valid_kind(kind):
        kind_data = results.get(kind)
        if not kind_data:
            return False
        sub_mask_results = kind_data.get("sub_mask_results", [])
        # "是多个" -> 数量大于 1
        if len(sub_mask_results) <= 1:
            return False
        # "有 object_unchanged == true"
        return any(sub.get("object_unchanged") is True for sub in sub_mask_results)

    if is_valid_kind("remove") and is_valid_kind("add"):
        return filepath
    
    return None

def main():
    # 确定搜索目录
    if os.path.exists(TARGET_DIR):
        search_dir = TARGET_DIR
    elif os.path.exists(ABS_TARGET_DIR):
        print(f"Using absolute path: {ABS_TARGET_DIR}")
        search_dir = ABS_TARGET_DIR
    else:
        print(f"Directory not found: {TARGET_DIR} or {ABS_TARGET_DIR}")
        return

    # 获取所有 JSON 文件
    print(f"Scanning directory: {search_dir} ...")
    pattern = os.path.join(search_dir, "*_dino_audit.json")
    json_files = glob.glob(pattern)
    total_files = len(json_files)
    print(f"Found {total_files} files. Starting processing with {cpu_count()} CPUs...")

    matched_files = []
    
    # 使用进程池并行处理
    # max_workers 默认是 cpu_count()，对于 IO 密集型也可以设大一点，但这里涉及 JSON 解析，CPU 也是瓶颈
    with ProcessPoolExecutor() as executor:
        # 提交所有任务
        futures = [executor.submit(check_file, f) for f in json_files]
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=total_files, unit="file"):
            result = future.result()
            if result:
                matched_files.append(result)

    print(f"\nProcessing complete.")
    print(f"Found {len(matched_files)} matching files:")
    
    # 打印结果（可选：限制打印数量，或者保存到文件）
    for f in matched_files:
        print(f)
        
    # 如果结果很多，保存到文件可能更好
    if matched_files:
        output_list_file = "matched_audit_files.txt"
        with open(output_list_file, "w") as f:
            for line in matched_files:
                f.write(f"{line}\n")
        print(f"\nList of matched files saved to {output_list_file}")

if __name__ == "__main__":
    main()
