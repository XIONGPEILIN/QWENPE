import json
import os

def main():
    # 加载数据
    if not os.path.exists("seam_quality_report.json") or not os.path.exists("dataset_with_audit_scores.json"):
        print("Error: Missing required score files.")
        return

    with open("seam_quality_report.json", "r") as f:
        seam_report = json.load(f)
        
    with open("dataset_with_audit_scores.json", "r") as f:
        audit_data = json.load(f)

    # 建立索引
    audit_map = {item["image"]: item.get("audit_bg_dist", 0.0) for item in audit_data}

    # 合并分数
    combined_results = []
    for entry in seam_report:
        img_path = entry["image"]
        seam_s = entry["seam_score"]
        # 有些样本可能没有审计分数，默认为 0
        audit_s = audit_map.get(img_path, 0.0)
        if audit_s is None: audit_s = 0.0
        
        # 综合分数逻辑：
        # seam_score 反应接缝硬度 (0~200+)
        # audit_bg_dist 反应背景差异感 (0~0.5+)
        # 乘积能有效放大那些“背景变了且接缝很生硬”的样本
        combined_score = seam_s * (audit_s + 0.05) 
        
        combined_results.append({
            "index": entry["index"],
            "image": img_path,
            "seam_score": seam_s,
            "audit_bg_dist": audit_s,
            "final_score": combined_score
        })

    # 按综合分数排序 (最差在前)
    combined_results.sort(key=lambda x: x["final_score"], reverse=True)

    with open("final_quality_ranking.json", "w") as f:
        json.dump(combined_results, f, indent=2)

    print("--- 综合质量最差 Top 10 ---")
    for i in range(10):
        res = combined_results[i]
        print(f"#{i+1}: FinalScore={res['final_score']:.2f} (Seam={res['seam_score']:.1f}, Audit={res['audit_bg_dist']:.3f}) - {res['image']}")

if __name__ == "__main__":
    main()
