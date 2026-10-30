"""
Products容量优化测试 - 分层推理
=====================================
目标: 解决Products的容量和速度问题，完成完整测试
当前: 未完成（速度2.44节点/秒，需要279小时）
论文: 81.84%
目标: 完成推理并达到79.8-83.8%范围

问题分析:
1. 2,449,029节点 × 1024维 ≈ 9.34GB
2. 当前速度: 2.44节点/秒 → 需要279小时
3. 内存压力: private 10.8GB

改进策略:
- 分层推理：复用中间层特征
- 临时存储：使用磁盘缓存
- 批量优化：增大batch size
- 预期速度提升3-5倍 → 总时间60-100小时

使用方法:
python scripts/test_03_products_layered.py --stage profile    # 性能分析
python scripts/test_03_products_layered.py --stage test_1k    # 测试前1000节点
python scripts/test_03_products_layered.py --stage full       # 完整推理
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import dgl
import psutil

# 添加项目路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.came_repro.author_loader import verify_originals, fingerprint
from src.came_repro.experiment import seed_everything
from src.came_repro.atomic_io import read_json_file


def profile_current_bottleneck():
    """分析当前的性能瓶颈"""
    print("=" * 80)
    print("阶段1: 性能瓶颈分析")
    print("=" * 80)

    print("\n📊 当前状况:")
    print(f"   节点数: 2,449,029")
    print(f"   边数: 126,167,053")
    print(f"   特征维度: 1024 (FP32)")
    print(f"   当前速度: 2.44 节点/秒")
    print(f"   预计总时间: 279 小时")
    print()

    # 检查系统资源
    print("💻 系统资源检查:")

    # CPU和内存
    cpu_count = psutil.cpu_count()
    mem = psutil.virtual_memory()

    print(f"   CPU核心: {cpu_count}")
    print(f"   总内存: {mem.total / (1024**3):.1f} GB")
    print(f"   可用内存: {mem.available / (1024**3):.1f} GB")
    print(f"   已用内存: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")

    # GPU
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"\n   GPU: {gpu_props.name}")
        print(f"   GPU内存: {gpu_props.total_memory / (1024**3):.1f} GB")

        # 当前GPU使用情况
        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU已分配: {mem_allocated:.2f} GB")
        print(f"   GPU已保留: {mem_reserved:.2f} GB")

    # 磁盘空间
    print(f"\n💾 磁盘空间:")
    for drive in ['C:', 'E:']:
        try:
            usage = psutil.disk_usage(drive)
            print(f"   {drive} 总空间: {usage.total / (1024**3):.1f} GB")
            print(f"   {drive} 可用: {usage.free / (1024**3):.1f} GB ({100 - usage.percent:.1f}%)")
        except:
            pass

    # 计算理论需求
    print(f"\n📐 理论资源需求:")

    num_nodes = 2449029
    feature_dim = 1024
    bytes_per_float32 = 4

    # 完整特征矩阵
    full_matrix_gb = num_nodes * feature_dim * bytes_per_float32 / (1024**3)
    print(f"   完整特征矩阵: {full_matrix_gb:.2f} GB")

    # 分层方案
    layer1_text_gb = full_matrix_gb
    layer1_vision_gb = full_matrix_gb
    layer2_fusion_gb = full_matrix_gb

    print(f"   分层方案临时存储:")
    print(f"   - Text层1: {layer1_text_gb:.2f} GB")
    print(f"   - Vision层1: {layer1_vision_gb:.2f} GB")
    print(f"   - 融合输出: {layer2_fusion_gb:.2f} GB")
    print(f"   - 峰值总计: {2 * full_matrix_gb:.2f} GB (2个临时矩阵)")

    # 检查C盘是否有足够空间
    c_usage = psutil.disk_usage('C:')
    c_free_gb = c_usage.free / (1024**3)
    needed_gb = 2 * full_matrix_gb + 3  # 额外3GB缓冲

    print(f"\n✓ 空间检查:")
    print(f"   需要: {needed_gb:.1f} GB")
    print(f"   可用: {c_free_gb:.1f} GB")

    if c_free_gb >= needed_gb:
        print(f"   ✅ 空间充足")
    else:
        print(f"   ⚠️  空间可能不足，建议清理磁盘")

    # 瓶颈分析
    print(f"\n🔍 瓶颈分析:")
    print(f"   1. 当前方案：逐节点完整2跳GAT推理")
    print(f"   2. 问题：重复计算邻居特征")
    print(f"   3. 改进：分层计算，复用中间结果")
    print()

    return {
        'num_nodes': num_nodes,
        'feature_dim': feature_dim,
        'needed_space_gb': needed_gb,
        'available_space_gb': c_free_gb
    }


def test_1k_nodes_speed():
    """测试前1000节点的速度和内存"""
    print("\n" + "=" * 80)
    print("阶段2: 小规模速度测试")
    print("=" * 80)

    print("\n🔧 测试配置:")
    print("   - 测试节点: 前1000个")
    print("   - Batch size: 64 (优化后)")
    print("   - 策略: 分层推理")
    print()

    # 检查是否有必要的输入
    cache_dir = ROOT / "data" / "processed" / "Products"
    encoder_file = ROOT / "runs" / "completed_joint_representation" / "encoder_final.pt"

    print("📋 检查依赖文件...")

    # 这里仅做模拟测试框架
    print("   ⚠️  实际推理需要:")
    print("   1. 冻结的编码器模型")
    print("   2. Products图结构")
    print("   3. CLIP文本和图像特征")
    print()

    print("💡 分层推理策略:")
    print()
    print("   第一层 (Text GAT):")
    print("   ├─ 输入: 原始text特征")
    print("   ├─ 输出: Text层1特征 → 存储到临时文件A")
    print("   └─ 释放输入内存")
    print()
    print("   第一层 (Vision GAT):")
    print("   ├─ 输入: 原始vision特征")
    print("   ├─ 输出: Vision层1特征 → 存储到临时文件B")
    print("   └─ 释放输入内存")
    print()
    print("   第二层 (Text GAT + 融合):")
    print("   ├─ 输入: 从文件A读取")
    print("   ├─ 输出: 融合特征 → 覆盖文件B的对应位置")
    print("   └─ 释放文件A")
    print()
    print("   第二层 (Vision GAT + 最终融合):")
    print("   ├─ 输入: 从文件A读取 (text已完成)")
    print("   ├─ 输入: 从文件B读取 (当前batch)")
    print("   ├─ 融合: Expert fusion")
    print("   └─ 输出: 最终1024维特征")
    print()

    # 估算性能提升
    print("📈 预期性能提升:")
    print("   - 当前: 2.44 节点/秒")
    print("   - 优化后: 7-12 节点/秒 (3-5倍提升)")
    print("   - 原因: 减少重复的邻居聚合计算")
    print()
    print("   总时间估算:")
    print("   - 悲观: 2,449,029 / 7 ≈ 97 小时")
    print("   - 乐观: 2,449,029 / 12 ≈ 57 小时")
    print()

    # 模拟小规模测试
    print("🧪 模拟前1000节点测试...")
    print("   (实际测试需要完整的模型和数据)")
    print()

    # 计算需要的时间
    test_nodes = 1000
    estimated_speed_range = (7, 12)

    time_pessimistic = test_nodes / estimated_speed_range[0]
    time_optimistic = test_nodes / estimated_speed_range[1]

    print(f"   1000节点预计时间:")
    print(f"   - 悲观: {time_pessimistic:.1f} 秒 ({time_pessimistic/60:.1f} 分钟)")
    print(f"   - 乐观: {time_optimistic:.1f} 秒 ({time_optimistic/60:.1f} 分钟)")
    print()

    print("✅ 小规模测试框架准备完成")
    print()

    return {
        'test_nodes': test_nodes,
        'estimated_time_seconds': (time_pessimistic, time_optimistic)
    }


def run_full_inference():
    """运行完整的分层推理"""
    print("\n" + "=" * 80)
    print("阶段3: 完整分层推理")
    print("=" * 80)

    print("\n⚠️  重要提示:")
    print("   这是一个长时间运行的任务（预计60-100小时）")
    print("   请确保:")
    print("   1. 电源连接")
    print("   2. 不要休眠/关机")
    print("   3. C盘有足够空间（约20GB）")
    print("   4. 可以断网运行")
    print()

    # 配置
    num_nodes = 2449029
    batch_size = 64
    feature_dim = 1024

    print(f"📊 配置:")
    print(f"   总节点: {num_nodes:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   特征维度: {feature_dim}")
    print()

    # 临时文件路径
    cache_base = Path("C:/Users/DELL/came_reproduction_cache")
    temp_dir = cache_base / "products_layered_inference"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 临时文件:")
    print(f"   目录: {temp_dir}")
    print(f"   - layer1_text.mmap ({9.34:.2f} GB)")
    print(f"   - layer1_vision.mmap ({9.34:.2f} GB)")
    print(f"   - final_fusion.mmap ({9.34:.2f} GB)")
    print()

    # 实际实现需要完整的推理代码
    print("📝 实现要点:")
    print("   1. 使用 np.memmap 创建磁盘映射数组")
    print("   2. 分4个阶段:")
    print("      - Text GAT Layer 1")
    print("      - Vision GAT Layer 1")
    print("      - Text GAT Layer 2 + Fusion")
    print("      - Vision GAT Layer 2 + Final Fusion")
    print("   3. 每个batch:")
    print("      - 采样邻居（DGL采样器）")
    print("      - GPU前向传播")
    print("      - 写回磁盘")
    print("   4. 进度保存:")
    print("      - cursor文件记录进度")
    print("      - 支持断点续推")
    print()

    print("🔄 执行流程:")
    print("   Phase 1/4: Text Layer 1")
    print("   ├─ 预计时间: 15-25小时")
    print("   └─ 输出: layer1_text.mmap")
    print()
    print("   Phase 2/4: Vision Layer 1")
    print("   ├─ 预计时间: 15-25小时")
    print("   └─ 输出: layer1_vision.mmap")
    print()
    print("   Phase 3/4: Text Layer 2 + Fusion")
    print("   ├─ 预计时间: 15-25小时")
    print("   └─ 输出: 覆盖部分layer1_vision.mmap")
    print()
    print("   Phase 4/4: Vision Layer 2 + Final")
    print("   ├─ 预计时间: 15-25小时")
    print("   └─ 输出: final_fusion.mmap")
    print()

    print("⏱️  总预计时间: 60-100小时")
    print()

    print("📋 完成后:")
    print("   1. 验证数值正确性（与小batch对比）")
    print("   2. 加载最终特征进行节点分类")
    print("   3. 在官方测试集上评测")
    print("   4. 目标准确率: 81.84±2% (79.84-83.84%)")
    print()

    return {
        'total_nodes': num_nodes,
        'estimated_hours': (60, 100),
        'temp_dir': str(temp_dir)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage',
        choices=['profile', 'test_1k', 'full'],
        required=True,
        help='执行阶段: profile(分析), test_1k(测试1000节点), full(完整推理)'
    )
    args = parser.parse_args()

    verify_originals()
    seed_everything(42)

    print(f"\n🚀 开始执行 Products分层推理优化测试")
    print(f"   阶段: {args.stage}")
    print(f"   目标: 完成推理并达到79.8-83.8%范围")
    print()

    if args.stage == 'profile':
        stats = profile_current_bottleneck()
        print("\n" + "=" * 80)
        print("✅ 性能分析完成！")
        print("=" * 80)
        print("\n📋 下一步:")
        print("   python scripts/test_03_products_layered.py --stage test_1k")

    elif args.stage == 'test_1k':
        result = test_1k_nodes_speed()
        print("\n" + "=" * 80)
        print("✅ 小规模测试框架准备完成！")
        print("=" * 80)
        print("\n📋 下一步:")
        print("   1. 实现完整的分层推理代码")
        print("   2. 先用1000节点验证速度和准确性")
        print("   3. 确认后运行完整推理:")
        print("      python scripts/test_03_products_layered.py --stage full")

    elif args.stage == 'full':
        config = run_full_inference()
        print("\n" + "=" * 80)
        print("✅ 完整推理配置完成！")
        print("=" * 80)
        print("\n⚠️  注意:")
        print("   实际推理需要完整实现分层推理代码")
        print("   预计运行时间: 60-100小时")
        print("   请确保系统稳定运行")


if __name__ == "__main__":
    main()
