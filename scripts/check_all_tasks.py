"""检查所有主任务完成状态"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 十个主任务及其论文目标
TASKS = {
    # 节点分类 (4个)
    "Ele-fashion": {"paper": 85.13, "path": "final_results/ele_fashion_ten_seeds/verified_results.json"},
    "Wiki-CS": {"paper": 78.57, "path": "final_results/wiki_cs_splits/verified_results.json"},
    "Arxiv": {"paper": 71.66, "path": "final_results/arxiv_ten_seeds/verified_results.json"},
    "Goodreads-NC": {"paper": 83.21, "path": "final_results/goodreads_nc_ten_seeds/verified_results.json"},

    # 链接预测 (3个)
    "Sports-LP": {"paper": 37.68, "path": "verified_job07_sports_delivery_repeats/results.json"},
    "Cloth-LP": {"paper": 28.88, "path": "cloth_final_repeats/results.json"},
    "Goodreads-LP": {"paper": 10.42, "path": "goodreads_filtered_repeats/results.json"},

    # 关系分类 (2个)
    "FB15K237": {"paper": 94.59, "path": "final_results/fb15k237_ten_seeds/verified_results.json"},
    "WN18RR": {"paper": 81.97, "path": "wn_final_repeats/results.json"},

    # 节点分类-大图 (1个)
    "Products": {"paper": 81.84, "path": "products_smart_inference/inference_result.json"},
}

print("="*80)
print("CAME 论文复现 - 十个主任务完成状态检查")
print("="*80)
print()

completed = []
incomplete = []

for task_name, info in TASKS.items():
    result_file = ROOT / "artifacts" / info["path"]

    if result_file.exists():
        data = json.loads(result_file.read_text(encoding='utf-8'))

        # 提取结果
        if "mean_test_pct" in data:
            mean = data["mean_test_pct"]
        elif "statistics" in data and "mean_pct" in data["statistics"]:
            mean = data["statistics"]["mean_pct"]
        elif "test_accuracy_pct" in data:
            mean = data["test_accuracy_pct"]
        else:
            mean = None

        paper = info["paper"]

        if mean is not None:
            gap = mean - paper
            status = "✅" if abs(gap) <= 2.0 else "⚠️"
            completed.append({
                "task": task_name,
                "mean": mean,
                "paper": paper,
                "gap": gap,
                "status": status
            })
            print(f"{status} {task_name:15s} 均值: {mean:6.2f}%  论文: {paper:6.2f}%  差距: {gap:+6.2f}pp")
        else:
            incomplete.append(task_name)
            print(f"❌ {task_name:15s} 结果文件存在但无法提取均值")
    else:
        incomplete.append(task_name)
        print(f"❌ {task_name:15s} 结果文件不存在")

print()
print("="*80)
print("汇总统计")
print("="*80)
print(f"完成任务: {len(completed)}/10")
print(f"未完成任务: {len(incomplete)}/10")
print()

if completed:
    达标 = sum(1 for t in completed if abs(t["gap"]) <= 2.0)
    print(f"达标任务 (±2pp以内): {达标}/{len(completed)}")
    print()

if incomplete:
    print("未完成任务列表:")
    for task in incomplete:
        print(f"  - {task}")
    print()

# 保存详细报告
report = {
    "total": 10,
    "completed": len(completed),
    "incomplete": len(incomplete),
    "within_2pp": sum(1 for t in completed if abs(t["gap"]) <= 2.0) if completed else 0,
    "completed_tasks": completed,
    "incomplete_tasks": incomplete
}

report_file = ROOT / "artifacts" / "all_tasks_status_report.json"
report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"详细报告已保存至: {report_file}")
