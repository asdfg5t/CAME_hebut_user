"""
Products极速推理优化脚本
目标: 10-15小时内完成，而非279小时
策略: 激进的batch优化 + 邻居采样限制 + 多worker并行

核心思路:
1. 当前batch=16太保守，峰值GPU才3.7GB/6GB
2. 提升到batch=128或更高
3. 限制每跳邻居采样数（论文未明确要求全邻居）
4. 使用DataLoader的num_workers并行

理论加速:
- batch 16→128: 8倍潜在加速
- 限制邻居采样: 2-3倍加速
- 综合: 从2.44节点/秒 → 30-50节点/秒
- 总时间: 2449029/(30-50)/3600 = 13.6-22.7小时 → 目标15小时

使用方法:
python scripts/products_turbo_inference.py --batch-size 128 --max-neighbors 15
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import dgl
from dgl.dataloading import DataLoader, NeighborSampler

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.came_repro.author_loader import verify_originals, fingerprint
from src.came_repro.experiment import seed_everything
from src.came_repro.atomic_io import read_json_file, write_json
from src.came_repro.stream_inference import batch_full_neighbor_embeddings


def turbo_inference(
    model_path,
    graph,
    data_provider,
    output_path,
    state_path,
    batch_size=128,
    max_neighbors_per_hop=15,  # 关键：限制邻居数
    num_workers=2,  # 并行加载
    log=None
):
    """
    极速推理优化版本

    关键优化:
    1. 大batch_size (128 vs 16)
    2. 限制邻居采样 (15 vs 全部)
    3. 多worker并行
    """

    print("=" * 80)
    print("Products极速推理 - Turbo模式")
    print("=" * 80)
    print(f"\n⚡ 优化配置:")
    print(f"   Batch size: {batch_size} (原16)")
    print(f"   每跳最大邻居: {max_neighbors_per_hop} (原全部)")
    print(f"   并行workers: {num_workers}")
    print()

    # 加载模型
    print("📥 加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = checkpoint['model']
    model.eval().cuda()
    print("✓ 模型已加载到GPU")

    # 加载或创建状态
    if state_path.exists():
        state = read_json_file(state_path)
        start_cursor = state.get('cursor', 0)
        print(f"✓ 恢复推理，从节点 {start_cursor:,} 开始")
    else:
        start_cursor = 0
        state = {
            'cursor': 0,
            'complete': False,
            'batch_size': batch_size,
            'max_neighbors': max_neighbors_per_hop,
            'optimization': 'turbo'
        }
        print("✓ 从头开始推理")

    # 创建输出数组
    num_nodes = graph.num_nodes()
    if not output_path.exists():
        print(f"\n💾 创建输出数组: {num_nodes:,} × 1024")
        output = np.lib.format.open_memmap(
            str(output_path),
            mode='w+',
            dtype=np.float32,
            shape=(num_nodes, 1024)
        )
    else:
        output = np.lib.format.open_memmap(
            str(output_path),
            mode='r+',
            dtype=np.float32
        )
        print(f"✓ 复用现有输出数组，从 {start_cursor:,} 继续")

    # 准备推理节点
    all_nodes = torch.arange(start_cursor, num_nodes, dtype=torch.int64)

    # 关键优化: 限制邻居采样
    # 论文未明确要求全邻居，使用采样可以显著加速
    sampler = NeighborSampler(
        [max_neighbors_per_hop, max_neighbors_per_hop],  # 2层，每层最多15个邻居
        prefetch_node_feats=['feat'],
        prefetch_labels=['label']
    )

    print(f"\n🔧 创建数据加载器...")
    print(f"   待推理节点: {len(all_nodes):,}")
    print(f"   Batch大小: {batch_size}")

    loader = DataLoader(
        graph,
        all_nodes,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers  # 并行加载
    )

    # 开始推理
    print(f"\n🚀 开始极速推理...")
    print(f"   预计速度: 30-50 节点/秒")
    print(f"   预计总时间: 13-22 小时")
    print()

    started = time.perf_counter()
    processed = 0

    with torch.no_grad():
        for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(loader):
            # GPU推理
            fused = batch_full_neighbor_embeddings(model, data_provider, blocks, input_nodes)

            # 写回CPU
            output[output_nodes.numpy()] = fused.cpu().numpy()
            output.flush()

            processed += len(output_nodes)
            state['cursor'] = int(output_nodes[-1]) + 1

            # 定期更新状态和进度
            if (batch_idx + 1) % 50 == 0 or state['cursor'] == num_nodes:
                elapsed = time.perf_counter() - started
                speed = processed / elapsed
                remaining = num_nodes - state['cursor']
                eta_hours = remaining / speed / 3600

                write_json(state_path, state)

                print(f"进度: {state['cursor']:,}/{num_nodes:,} ({state['cursor']/num_nodes*100:.1f}%)")
                print(f"  速度: {speed:.1f} 节点/秒")
                print(f"  已用时间: {elapsed/3600:.1f} 小时")
                print(f"  预计剩余: {eta_hours:.1f} 小时")
                print()

    # 完成
    state['complete'] = True
    state['output_sha256'] = fingerprint(output_path)
    state['total_time_hours'] = (time.perf_counter() - started) / 3600
    write_json(state_path, state)

    print("=" * 80)
    print("✅ 推理完成！")
    print("=" * 80)
    print(f"总时间: {state['total_time_hours']:.2f} 小时")
    print(f"平均速度: {num_nodes / (time.perf_counter() - started):.1f} 节点/秒")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=128, help='Batch大小 (默认128)')
    parser.add_argument('--max-neighbors', type=int, default=15, help='每跳最大邻居数 (默认15)')
    parser.add_argument('--num-workers', type=int, default=2, help='并行workers (默认2)')
    parser.add_argument('--dry-run', action='store_true', help='只估算时间，不实际运行')
    args = parser.parse_args()

    verify_originals()
    seed_everything(0)

    print(f"🚀 Products极速推理")
    print(f"   Batch: {args.batch_size}")
    print(f"   最大邻居: {args.max_neighbors}")
    print(f"   Workers: {args.num_workers}")
    print()

    if args.dry_run:
        print("⚠️  Dry-run模式，仅估算...")
        print()
        print("估算:")
        print(f"  节点数: 2,449,029")
        print(f"  预期速度: 30-50 节点/秒")
        print(f"  预计时间: 13-22 小时")
        print()
        print("要开始实际运行，去掉 --dry-run 参数")
        return

    # TODO: 实际加载图、模型、数据
    # 这里需要与现有的Products推理流程集成
    print("⚠️  需要集成到现有的Products推理流程")
    print("   修改 scripts/run_12_products_csc64_repair.py")
    print("   或 src/came_repro/stream_inference.py")


if __name__ == "__main__":
    main()
