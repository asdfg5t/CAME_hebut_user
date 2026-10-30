"""
WN18RR改进测试 - OFA文本前缀增强
=======================================
目标: 验证OFA官方文本前缀是否能改善WN18RR性能
当前: 78.13% (10 seeds), 论文: 81.97%, 差距: -3.84pp
预期: seed=0 validation提升≥0.5pp (从80.29%到80.79%+)

科学依据:
- 论文附录C明确使用OFA数据
- OFA data/KG/gen_data.py:84 添加任务无关前缀
- 格式: "Entity: {name}. Description: {description}"

使用方法:
python scripts/test_01_wn_ofa_improvement.py --stage prepare  # 生成新特征
python scripts/test_01_wn_ofa_improvement.py --stage validate # 训练seed=0
python scripts/test_01_wn_ofa_improvement.py --stage test     # 确认测试
"""
import argparse
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# 添加项目路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.came_repro.author_loader import verify_originals
from src.came_repro.experiment import seed_everything


def generate_ofa_text_features():
    """生成带OFA前缀的WN18RR文本特征"""
    print("=" * 80)
    print("阶段1: 生成OFA前缀文本特征")
    print("=" * 80)

    # 1. 加载WN18RR实体文本
    data_dir = ROOT / "data" / "prepared" / "WN18RR"
    entity_text_file = data_dir / "entity2text.txt"

    if not entity_text_file.exists():
        raise FileNotFoundError(f"实体文本文件不存在: {entity_text_file}")

    print(f"\n📖 读取实体文本: {entity_text_file}")
    with open(entity_text_file, 'r', encoding='utf-8') as f:
        entity_texts = [line.strip() for line in f]

    num_entities = len(entity_texts)
    print(f"✓ 实体数量: {num_entities:,}")

    # 2. 应用OFA前缀格式
    print("\n🔧 应用OFA前缀格式...")
    print("   格式: 'Entity: {name}. Description: {description}'")

    ofa_texts = []
    for i, text in enumerate(entity_texts):
        # OFA格式: 添加任务无关前缀
        # 提取实体名称（通常是第一个词或整个短语）
        parts = text.split('.', 1)
        if len(parts) == 2:
            entity_name = parts[0].strip()
            description = parts[1].strip()
            ofa_text = f"Entity: {entity_name}. Description: {description}"
        else:
            # 如果没有描述，只添加Entity前缀
            ofa_text = f"Entity: {text}"

        ofa_texts.append(ofa_text)

        # 显示前3个示例
        if i < 3:
            print(f"   原文: {text[:60]}...")
            print(f"   OFA:  {ofa_text[:60]}...")
            print()

    # 3. 加载CLIP模型
    print("📥 加载CLIP模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"✓ 模型加载完成 (设备: {device})")

    # 4. 生成CLIP特征
    print("\n🔄 生成CLIP文本特征...")
    batch_size = 256
    all_features = []

    with torch.no_grad():
        for i in range(0, num_entities, batch_size):
            batch_texts = ofa_texts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            # 提取文本特征
            outputs = model(**inputs)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            # L2归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_features.append(text_features.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                print(f"   进度: {i:,} / {num_entities:,}")

    # 5. 合并并保存
    all_features = np.vstack(all_features)
    print(f"\n✓ 特征生成完成")
    print(f"   形状: {all_features.shape}")
    print(f"   数据类型: {all_features.dtype}")

    # 保存特征
    output_dir = ROOT / "data" / "processed" / "WN18RR_ofa_text"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "clip_text_features.npy"
    np.save(output_file, all_features.astype(np.float32))

    # 计算SHA256
    with open(output_file, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    # 保存元数据
    metadata = {
        "source": "OFA_prefix_enhanced",
        "model": model_name,
        "num_entities": num_entities,
        "feature_dim": 512,
        "sha256": sha256,
        "format": "Entity: {name}. Description: {description}",
        "original_text_file": str(entity_text_file)
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 特征已保存:")
    print(f"   文件: {output_file}")
    print(f"   SHA256: {sha256}")
    print(f"   元数据: {metadata_file}")
    print(f"\n✅ 阶段1完成！")

    return output_file, metadata


def validate_seed0():
    """使用新特征训练seed=0并验证"""
    print("\n" + "=" * 80)
    print("阶段2: seed=0验证训练")
    print("=" * 80)

    print("\n⚠️  重要提示:")
    print("   1. 此阶段需要重新编码整个WN18RR图")
    print("   2. 使用新的OFA文本特征")
    print("   3. 其他设置保持不变:")
    print("      - train_reverse图政策")
    print("      - train_pair_affine归一化")
    print("      - head-tail拼接线性分类器")
    print("      - Adam lr=0.05, max_epochs=2000, patience=100")
    print()

    # 检查新特征是否存在
    feature_dir = ROOT / "data" / "processed" / "WN18RR_ofa_text"
    feature_file = feature_dir / "clip_text_features.npy"

    if not feature_file.exists():
        print("❌ 错误: 新特征文件不存在")
        print("   请先运行: python scripts/test_01_wn_ofa_improvement.py --stage prepare")
        return False

    print(f"✓ 找到新特征文件: {feature_file}")

    # 提示需要运行完整的编码和训练流程
    print("\n📝 下一步操作:")
    print("   1. 需要运行完整的图编码流程（使用新文本特征）")
    print("   2. 训练seed=0的关系分类头")
    print("   3. 比较validation准确率:")
    print("      - 原版: 80.29%")
    print("      - 目标: ≥80.79% (+0.5pp)")
    print()
    print("   由于这需要调用现有的训练流程，建议:")
    print("   - 参考 scripts/run_18_wn_final_repeats.py")
    print("   - 修改特征加载路径指向新特征")
    print("   - 运行单seed训练")
    print()

    return True


def test_seed0():
    """确认seed=0测试结果"""
    print("\n" + "=" * 80)
    print("阶段3: seed=0测试确认")
    print("=" * 80)

    print("\n📊 验收标准:")
    print("   - 如果validation ≥ 80.79%: 进行一次完整test")
    print("   - 如果test进入81.97±2pp (79.97-83.97%): 投入10 seeds完整训练")
    print("   - 如果未达标: 保留结果，不继续")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage',
        choices=['prepare', 'validate', 'test'],
        required=True,
        help='执行阶段: prepare(生成特征), validate(训练验证), test(测试确认)'
    )
    args = parser.parse_args()

    verify_originals()
    seed_everything(42)

    print(f"\n🚀 开始执行 WN18RR OFA文本改进测试")
    print(f"   阶段: {args.stage}")
    print(f"   目标: 从78.13%提升至80.0-84.0%范围")
    print()

    if args.stage == 'prepare':
        generate_ofa_text_features()
        print("\n" + "=" * 80)
        print("✅ 特征生成完成！")
        print("=" * 80)
        print("\n📋 下一步:")
        print("   python scripts/test_01_wn_ofa_improvement.py --stage validate")

    elif args.stage == 'validate':
        validate_seed0()

    elif args.stage == 'test':
        test_seed0()


if __name__ == "__main__":
    main()
