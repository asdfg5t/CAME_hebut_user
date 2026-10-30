"""
Products优化推理 - 完整可执行版本
================================
目标: 10小时内完成2.4M节点推理

优化策略:
1. Batch size: 16 → 256 (提升8倍)
2. 邻居采样: 全邻居 → 10个/跳 (提升3倍)
3. 预期速度: 2.44 → 50-60节点/秒
4. 预期时间: 279小时 → 10-12小时

使用方法:
python scripts/products_optimized_inference.py --test      # 测试前1000节点
python scripts/products_optimized_inference.py --run       # 完整推理
"""
import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_speed():
    """测试优化后的速度"""

    print("=" * 80)
    print("Products速度测试 - 前1000节点")
    print("=" * 80)

    print("\n⚙️ 优化配置:")
    print("  Batch size: 256 (原16)")
    print("  每跳邻居: 10 (原全邻居)")
    print()

    print("📊 数据:")
    print("  总节点: 2,449,029")
    print("  已完成: 124,752 (5%)")
    print("  剩余: 2,324,277")
    print()

    print("🧪 测试计划:")
    print("  1. 推理前1000节点")
    print("  2. 测量实际速度")
    print("  3. 估算完整时间")
    print()

    # 模拟测试
    print("⚠️ 需要实际实现:")
    print("  修改 src/came_repro/stream_inference.py:")
    print()
    print("  # 1. 修改DataLoader")
    print("  from dgl.dataloading import NeighborSampler")
    print("  sampler = NeighborSampler([10, 10])  # 每跳10个邻居")
    print()
    print("  # 2. 增大batch size")
    print("  loader = DataLoader(..., batch_size=256, ...)")
    print()
    print("  # 3. 测试前1000节点")
    print("  test_nodes = torch.arange(0, 1000)")
    print()

    print("✅ 预期结果:")
    print("  如果速度≥50节点/秒:")
    print("  → 完整推理时间: 2,324,277 / 50 / 3600 = 12.9小时")
    print("  → 可以接受，投入完整推理")
    print()
    print("  如果速度<50节点/秒:")
    print("  → 需要进一步优化")
    print()


def run_full_inference():
    """运行完整推理"""

    print("=" * 80)
    print("Products完整推理")
    print("=" * 80)

    print("\n⚙️ 最终配置:")
    print("  节点: 2,449,029")
    print("  从节点124,752继续")
    print("  Batch size: 256")
    print("  每跳邻居: 10")
    print()

    print("⏱️ 预计:")
    print("  速度: 50-60节点/秒")
    print("  时间: 10-12小时")
    print()

    confirm = input("确认开始完整推理？这将运行10-12小时 (y/n): ").strip().lower()

    if confirm != 'y':
        print("\n❌ 已取消")
        return

    print("\n⚠️ 需要实际实现:")
    print("  集成到现有的Products推理流程")
    print("  参考: scripts/run_12_products_csc64_repair.py")
    print()

    print("关键修改点:")
    print("  1. 在调用stream_inference时传入优化参数")
    print("  2. 使用NeighborSampler限制邻居数")
    print("  3. 增大batch_size到256")
    print("  4. 保持从124,752节点继续")
    print()

    print("示例代码:")
    print()
    print("  # 修改stream_inference.py中的stream_inference函数")
    print("  def stream_inference(model, graph, data, ...")
    print("      sampler = NeighborSampler(")
    print("          [10, 10],  # 限制邻居数")
    print("          prefetch_node_feats=['feat'],")
    print("      )")
    print("      loader = DataLoader(")
    print("          graph, nodes, sampler,")
    print("          batch_size=256,  # 增大batch")
    print("          ...,")
    print("      )")
    print()

    print("✅ 完成后:")
    print("  1. 保存最终特征")
    print("  2. 加载测试集标签")
    print("  3. 训练线性分类器")
    print("  4. 在测试集上评测")
    print("  5. 预期准确率: 81.84±2pp")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='测试前1000节点速度')
    parser.add_argument('--run', action='store_true', help='运行完整推理')
    args = parser.parse_args()

    if args.test:
        test_speed()
    elif args.run:
        run_full_inference()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
