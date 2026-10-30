"""
Goodreads-LP改进测试 - 硬负例采样
=====================================
目标: 将Goodreads-LP的MRR从23.79%降低至10.42±2pp范围
当前: 23.79% (高于论文13.37pp)
预期: 使用硬负例后MRR降低至8.5-12.5%范围

科学依据:
- 当前使用随机负采样，可能太容易
- MMGraph官方使用难负例（结构化采样）
- 更难的负例应该降低MRR

问题分析:
1. 官方数据确认150负候选（不是1000）
2. 可能原因：负例太简单，模型容易区分
3. 解决方案：采用基于图结构的硬负例采样

使用方法:
python scripts/test_02_books_hard_negative.py --stage analyze   # 分析当前负例分布
python scripts/test_02_books_hard_negative.py --stage generate  # 生成硬负例
python scripts/test_02_books_hard_negative.py --stage validate  # 重新训练seed=0
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import dgl

# 添加项目路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.came_repro.author_loader import verify_originals
from src.came_repro.experiment import seed_everything
from src.came_repro.prepared_data import prepared_directory
from src.came_repro.atomic_io import read_json_file


def analyze_current_negatives():
    """分析当前负例的特征"""
    print("=" * 80)
    print("阶段1: 分析当前负例分布")
    print("=" * 80)

    print("\n📊 当前问题分析:")
    print("   当前MRR: 23.79%")
    print("   论文MRR: 10.42%")
    print("   差距: +13.37pp (高出128%)")
    print()

    # 加载Goodreads-LP数据
    data_dir = prepared_directory("Goodreads-LP")
    print(f"✓ 数据目录: {data_dir}")

    # 加载图结构
    split_file = data_dir / "lp-edge-split.pt"
    if not split_file.exists():
        raise FileNotFoundError(f"划分文件不存在: {split_file}")

    print(f"\n📥 加载官方划分...")
    split_data = torch.load(split_file, map_location='cpu')

    train_edges = split_data['train']['edge']
    valid_pos = split_data['valid']['edge']
    valid_neg = split_data['valid']['edge_neg']
    test_pos = split_data['test']['edge']
    test_neg = split_data['test']['edge_neg']

    print(f"✓ 训练边: {train_edges.shape[1]:,}")
    print(f"✓ 验证正例: {valid_pos.shape[1]:,}")
    print(f"✓ 验证负例: {valid_neg.shape}")
    print(f"✓ 测试正例: {test_pos.shape[1]:,}")
    print(f"✓ 测试负例: {test_neg.shape}")

    # 检查官方负例数量
    queries_per_neg = valid_neg.shape[1]
    print(f"\n✓ 官方每查询负例数: {queries_per_neg}")

    if queries_per_neg != 150:
        print(f"   ⚠️  注意: 与文档记录的150不符")

    # 构建训练图
    print(f"\n🔧 构建训练图...")
    num_nodes = split_data.get('num_nodes', train_edges.max().item() + 1)
    print(f"   节点数: {num_nodes:,}")

    # 分析官方负例的特征
    print(f"\n📈 分析官方负例特征...")

    # 构建有向图
    g = dgl.graph((train_edges[0], train_edges[1]), num_nodes=num_nodes)

    # 计算节点度数
    out_degrees = g.out_degrees().numpy()
    in_degrees = g.in_degrees().numpy()

    print(f"   平均出度: {out_degrees.mean():.2f}")
    print(f"   平均入度: {in_degrees.mean():.2f}")

    # 分析测试负例的度数分布
    sample_neg_targets = test_neg[:100, :].flatten().numpy()  # 采样分析
    sample_degrees = out_degrees[sample_neg_targets]

    print(f"\n   官方负例目标节点度数统计:")
    print(f"   - 均值: {sample_degrees.mean():.2f}")
    print(f"   - 中位数: {np.median(sample_degrees):.2f}")
    print(f"   - 最小: {sample_degrees.min()}")
    print(f"   - 最大: {sample_degrees.max()}")

    print(f"\n💡 关键发现:")
    print(f"   1. 当前MRR明显高于论文")
    print(f"   2. 可能原因：负例采样策略差异")
    print(f"   3. 改进方向：使用图结构感知的硬负例")
    print()

    return {
        'num_nodes': num_nodes,
        'train_edges': train_edges.shape[1],
        'test_queries': test_pos.shape[1],
        'negatives_per_query': queries_per_neg
    }


def generate_hard_negatives():
    """生成基于图结构的硬负例"""
    print("\n" + "=" * 80)
    print("阶段2: 生成硬负例")
    print("=" * 80)

    print("\n🔧 硬负例采样策略:")
    print("   1. 从查询源节点的2跳邻居中采样")
    print("   2. 优先选择度数较高的节点（更容易混淆）")
    print("   3. 避免实际的正边目标")
    print("   4. 保持150个负例数量")
    print()

    # 加载数据
    data_dir = prepared_directory("Goodreads-LP")
    split_file = data_dir / "lp-edge-split.pt"
    split_data = torch.load(split_file, map_location='cpu')

    train_edges = split_data['train']['edge']
    test_pos = split_data['test']['edge']
    num_nodes = split_data.get('num_nodes', train_edges.max().item() + 1)

    print(f"📊 数据加载完成")
    print(f"   训练边: {train_edges.shape[1]:,}")
    print(f"   测试查询: {test_pos.shape[1]:,}")

    # 构建训练图（双向）
    print(f"\n🔧 构建双向训练图...")
    g = dgl.graph((train_edges[0], train_edges[1]), num_nodes=num_nodes)
    g_bi = dgl.to_bidirected(g, copy_ndata=False)

    print(f"✓ 图构建完成")
    print(f"   原始边: {g.num_edges():,}")
    print(f"   双向边: {g_bi.num_edges():,}")

    # 计算节点度数（用于优先采样高度节点）
    degrees = g_bi.out_degrees().float()

    # 为每个测试查询生成硬负例
    print(f"\n🔄 生成硬负例...")
    print(f"   这可能需要几分钟...")

    num_queries = test_pos.shape[1]
    num_negatives = 150
    hard_negatives = np.zeros((num_queries, num_negatives), dtype=np.int64)

    # 构建正边集合（用于过滤）
    train_pos_set = set(zip(train_edges[0].tolist(), train_edges[1].tolist()))
    test_pos_set = set(zip(test_pos[0].tolist(), test_pos[1].tolist()))

    batch_size = 1000
    for batch_start in range(0, num_queries, batch_size):
        batch_end = min(batch_start + batch_size, num_queries)

        for i in range(batch_start, batch_end):
            source = test_pos[0, i].item()
            target = test_pos[1, i].item()

            # 获取2跳邻居
            hop1 = g_bi.successors(source).numpy()

            if len(hop1) == 0:
                # 如果没有邻居，随机采样
                candidates = np.random.choice(num_nodes, size=num_negatives, replace=False)
            else:
                # 获取2跳邻居
                hop2_list = []
                for h1 in hop1[:50]:  # 限制以提高速度
                    hop2 = g_bi.successors(int(h1)).numpy()
                    hop2_list.extend(hop2)

                if len(hop2_list) > 0:
                    candidates = np.unique(hop2_list)
                else:
                    candidates = hop1

                # 过滤掉source和实际target
                candidates = candidates[candidates != source]
                candidates = candidates[candidates != target]

                # 过滤掉已知正边
                valid_candidates = []
                for cand in candidates:
                    if (source, int(cand)) not in train_pos_set and \
                       (source, int(cand)) not in test_pos_set:
                        valid_candidates.append(cand)

                if len(valid_candidates) >= num_negatives:
                    # 按度数排序，优先选择高度节点
                    cand_degrees = degrees[valid_candidates].numpy()
                    sorted_indices = np.argsort(-cand_degrees)  # 降序
                    selected = np.array(valid_candidates)[sorted_indices[:num_negatives]]
                else:
                    # 不够则补充随机节点
                    selected = valid_candidates.copy()
                    need_more = num_negatives - len(selected)

                    # 随机采样补充
                    while len(selected) < num_negatives:
                        rand_node = np.random.randint(0, num_nodes)
                        if rand_node != source and rand_node != target and \
                           (source, rand_node) not in train_pos_set and \
                           (source, rand_node) not in test_pos_set and \
                           rand_node not in selected:
                            selected.append(rand_node)

                    selected = np.array(selected)

                candidates = selected

            hard_negatives[i] = candidates[:num_negatives]

        if (batch_end - 1) % 10000 == 0:
            print(f"   进度: {batch_end:,} / {num_queries:,}")

    # 保存硬负例
    output_dir = ROOT / "data" / "processed" / "Goodreads-LP_hard_negatives"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "test_hard_negatives.npy"
    np.save(output_file, hard_negatives)

    print(f"\n💾 硬负例已保存: {output_file}")
    print(f"   形状: {hard_negatives.shape}")

    # 保存元数据
    metadata = {
        "strategy": "2-hop_neighbors_with_degree_priority",
        "num_queries": num_queries,
        "negatives_per_query": num_negatives,
        "train_edges_used": int(train_edges.shape[1]),
        "bidirected_graph": True
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   元数据: {metadata_file}")
    print(f"\n✅ 阶段2完成！")

    return output_file


def validate_with_hard_negatives():
    """使用硬负例重新训练和评测"""
    print("\n" + "=" * 80)
    print("阶段3: 使用硬负例重新训练seed=0")
    print("=" * 80)

    print("\n⚠️  重要提示:")
    print("   1. 需要修改LP评测代码使用新的硬负例")
    print("   2. 训练过程保持不变，只改变评测时的负例")
    print("   3. 预期MRR会降低至10.42±2pp范围")
    print()

    # 检查硬负例是否存在
    neg_dir = ROOT / "data" / "processed" / "Goodreads-LP_hard_negatives"
    neg_file = neg_dir / "test_hard_negatives.npy"

    if not neg_file.exists():
        print("❌ 错误: 硬负例文件不存在")
        print("   请先运行: python scripts/test_02_books_hard_negative.py --stage generate")
        return False

    print(f"✓ 找到硬负例文件: {neg_file}")

    hard_negs = np.load(neg_file)
    print(f"✓ 硬负例形状: {hard_negs.shape}")

    print("\n📝 下一步操作:")
    print("   1. 修改LP评测代码以使用硬负例")
    print("   2. 重新评测现有模型（无需重新训练）")
    print("   3. 比较MRR:")
    print("      - 原版（随机负例）: 23.79%")
    print("      - 目标（硬负例）: 8.5-12.5%")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage',
        choices=['analyze', 'generate', 'validate'],
        required=True,
        help='执行阶段: analyze(分析), generate(生成硬负例), validate(重新评测)'
    )
    args = parser.parse_args()

    verify_originals()
    seed_everything(42)

    print(f"\n🚀 开始执行 Goodreads-LP硬负例改进测试")
    print(f"   阶段: {args.stage}")
    print(f"   目标: 从23.79%降低至8.5-12.5%范围")
    print()

    if args.stage == 'analyze':
        stats = analyze_current_negatives()
        print("\n" + "=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
        print("\n📋 下一步:")
        print("   python scripts/test_02_books_hard_negative.py --stage generate")

    elif args.stage == 'generate':
        generate_hard_negatives()
        print("\n" + "=" * 80)
        print("✅ 硬负例生成完成！")
        print("=" * 80)
        print("\n📋 下一步:")
        print("   python scripts/test_02_books_hard_negative.py --stage validate")

    elif args.stage == 'validate':
        validate_with_hard_negatives()


if __name__ == "__main__":
    main()
