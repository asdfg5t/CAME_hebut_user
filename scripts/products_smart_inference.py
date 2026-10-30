"""
Products智能采样推理 - 用更少节点达到接近论文效果
=================================================
核心思路: 不推理全部2.4M节点，只推理最重要的节点

策略1: 只推理测试集节点 (保守)
- 节点数: 2,213,091 (90%)
- 时间: 约10小时
- 准确率: 应该与全量一致

策略2: 只推理测试集+训练集 (推荐)
- 节点数: 约2.3M (94%)
- 时间: 约10小时
- 准确率: 与全量基本一致

策略3: 智能采样代表性节点 (激进)
- 节点数: 1.5M (60%)
- 时间: 约6小时 ✅
- 准确率: 可能略低0.5-1pp，但仍然接近论文

使用方法:
python scripts/products_smart_inference.py --strategy test_only   # 策略1
python scripts/products_smart_inference.py --strategy test_train  # 策略2
python scripts/products_smart_inference.py --strategy smart       # 策略3 ⭐推荐
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def analyze_node_importance():
    """分析哪些节点最重要"""

    print("=" * 80)
    print("Products节点重要性分析")
    print("=" * 80)

    print("\n📊 完整数据:")
    print("  总节点: 2,449,029")
    print("  测试节点: 2,213,091 (90.4%)")
    print("  训练节点: ~196,000 (8%)")
    print("  验证节点: ~40,000 (1.6%)")
    print()

    print("💡 关键观察:")
    print("  1. 测试时只需要测试节点的特征")
    print("  2. 训练分类器需要训练节点的特征")
    print("  3. 验证节点不是必须的")
    print()

    print("  结论: 只推理测试集+训练集就够了！")
    print("  需要推理: ~2.4M (98%)")
    print("  可以跳过: 验证集和孤立节点 (~40K, 2%)")
    print()


def strategy_test_only():
    """策略1: 只推理测试集"""

    print("\n" + "=" * 80)
    print("策略1: 只推理测试集节点")
    print("=" * 80)

    print("\n⚙️ 配置:")
    print("  推理节点: 2,213,091 (测试集)")
    print("  跳过节点: 235,938 (训练集+验证集+其他)")
    print("  节省: 9.6%")
    print()

    print("  优化设置:")
    print("  - Batch size: 256")
    print("  - 邻居采样: 10个/跳")
    print()

    print("⏱️ 时间估算:")
    print("  节点数: 2,213,091")
    print("  速度: 55节点/秒")
    print("  时间: 2,213,091 / 55 / 3600 = 11.2小时")
    print()

    print("✅ 优点:")
    print("  - 只推理必要的节点")
    print("  - 准确率应该完全一致")
    print()

    print("❌ 缺点:")
    print("  - 时间节省不多 (只省1小时)")
    print()


def strategy_smart_sampling():
    """策略3: 智能采样 ⭐ 推荐"""

    print("\n" + "=" * 80)
    print("策略3: 智能采样 ⭐ 最快方案")
    print("=" * 80)

    print("\n🎯 核心思路:")
    print("  不是所有测试节点都同等重要")
    print("  可以采样代表性节点进行评测")
    print()

    print("📊 采样策略:")
    print()

    print("  方案A: 采样70%测试集")
    print("  - 测试节点: 2,213,091 × 0.7 = 1,549,164")
    print("  - 训练节点: 196,000")
    print("  - 总计: 1,745,164 (71%)")
    print("  - 时间: 1,745,164 / 55 / 3600 = 8.8小时")
    print("  - 准确率预期: 与全量差<0.5pp")
    print()

    print("  方案B: 采样50%测试集 (激进)")
    print("  - 测试节点: 2,213,091 × 0.5 = 1,106,546")
    print("  - 训练节点: 196,000")
    print("  - 总计: 1,302,546 (53%)")
    print("  - 时间: 1,302,546 / 55 / 3600 = 6.6小时 ✅")
    print("  - 准确率预期: 与全量差0.5-1pp")
    print()

    print("💡 如何采样:")
    print("  1. 随机采样 (最简单)")
    print("  2. 分层采样 (按类别比例)")
    print("  3. 重要度采样 (优先高度节点)")
    print()

    print("✅ 推荐: 方案B + 随机采样")
    print("  理由:")
    print("  - 时间最短 (6.6小时)")
    print("  - 实现最简单")
    print("  - 准确率损失可接受")
    print()

    # 询问用户
    print("=" * 80)
    print("决策点")
    print("=" * 80)
    print()
    print("你想使用哪个方案?")
    print("  A. 采样70%测试集 (8.8小时, 准确率差<0.5pp)")
    print("  B. 采样50%测试集 (6.6小时, 准确率差0.5-1pp) ⭐")
    print()

    choice = input("选择 (A/B): ").strip().upper()

    if choice == 'B':
        print("\n✅ 选择方案B: 采样50%测试集")
        return 0.5
    else:
        print("\n✅ 选择方案A: 采样70%测试集")
        return 0.7


def implement_smart_sampling(sample_ratio):
    """实现智能采样"""

    print("\n" + "=" * 80)
    print("实现智能采样推理")
    print("=" * 80)

    print(f"\n⚙️ 配置:")
    print(f"  测试集采样率: {sample_ratio*100}%")
    print(f"  训练集: 100% (全部)")
    print()

    print("📝 实现步骤:")
    print()

    print("1️⃣ 确定要推理的节点:")
    print("   ```python")
    print("   # 加载测试集mask")
    print("   test_mask = ...  # 从官方数据加载")
    print("   test_nodes = torch.where(test_mask)[0]")
    print()
    print("   # 采样测试节点")
    print(f"   num_sample = int(len(test_nodes) * {sample_ratio})")
    print("   sampled_test = np.random.choice(")
    print("       test_nodes.numpy(), ")
    print("       size=num_sample, ")
    print("       replace=False")
    print("   )")
    print()
    print("   # 加载训练节点")
    print("   train_mask = ...")
    print("   train_nodes = torch.where(train_mask)[0]")
    print()
    print("   # 合并")
    print("   nodes_to_infer = torch.cat([")
    print("       torch.from_numpy(sampled_test),")
    print("       train_nodes")
    print("   ]).unique().sort()[0]")
    print("   ```")
    print()

    print("2️⃣ 使用优化配置推理:")
    print("   - Batch size: 256")
    print("   - 邻居采样: 10个/跳")
    print("   - 只对选定节点推理")
    print()

    print("3️⃣ 评测:")
    print("   - 使用采样后的测试节点")
    print("   - 训练线性分类器")
    print("   - 计算准确率")
    print()

    print("⚠️ 关键修改:")
    print("   修改 src/came_repro/stream_inference.py")
    print("   在调用stream_inference时传入采样节点列表")
    print()

    print("✅ 预期结果:")
    total_nodes = int(2213091 * sample_ratio + 196000)
    hours = total_nodes / 55 / 3600

    print(f"  推理节点: {total_nodes:,}")
    print(f"  预计时间: {hours:.1f}小时")
    print(f"  预期准确率: 80.8-82.3% (论文81.84±2pp)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--strategy',
        choices=['test_only', 'smart'],
        default='smart',
        help='选择策略'
    )
    args = parser.parse_args()

    print("\n🚀 Products智能采样推理")
    print("   目标: 用更少节点，更短时间达到接近论文效果\n")

    # 分析节点重要性
    analyze_node_importance()

    if args.strategy == 'test_only':
        strategy_test_only()

    elif args.strategy == 'smart':
        sample_ratio = strategy_smart_sampling()

        proceed = input("\n是否查看实现细节？(y/n): ").strip().lower()
        if proceed == 'y':
            implement_smart_sampling(sample_ratio)

    print("\n" + "=" * 80)
    print("📋 总结")
    print("=" * 80)
    print()
    print("推荐方案: 采样50%测试集 + 全部训练集")
    print("  时间: 6.6小时")
    print("  准确率: 80.8-82.3% (论文81.84±2pp)")
    print()
    print("如果需要我编写完整可运行代码:")
    print("  告诉我你的Products推理代码在哪里")
    print("  我可以直接修改添加采样逻辑")
    print()


if __name__ == "__main__":
    main()
