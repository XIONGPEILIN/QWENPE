import json
import sys

def convert_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    flat_data = []
    
    # Iterate through all tasks in the "data" dictionary
    for task_name, items in data.get('data', {}).items():
        for item in items:
            flat_item = {
                "prompt": item.get("Instruction", ""),
                "edit_image": [item.get("SourceImage", "")],
                "back_mask": item.get("SourceMask", ""),
                "item_id": item.get("ItemID", ""),
                "task": item.get("Task", "")
            }
            flat_data.append(flat_item)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flat_data, f, indent=2, ensure_ascii=False)
        
    print(f"转换完成！共处理了 {len(flat_data)} 条数据。")
    print(f"结果已保存到：{output_path}")

if __name__ == "__main__":
    input_file = "ICE-Bench/dataset/selected_5_tasks.json"
    output_file = "ICE-Bench/dataset/selected_5_tasks_flat.json"
    
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
    convert_format(input_file, output_file)
