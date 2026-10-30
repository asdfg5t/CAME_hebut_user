# 第二阶段：正式复现的数据与协议准备

开始日期：2026-10-10（Asia/Shanghai）。前一阶段的 pilot 结果保留，不用于宣称完整论文性能。

## 目标与来源

目标是补齐两个正式预训练语料、核实 CLIP 特征语义，并建立能覆盖论文主任务的训练与评测入口。依据为用户论文第 6 页 §5.2、第 14 页附录 C、官方 MM-Graph、OGB 和论文引用的 OFA/UniGraph2。

重新核对 §5.2 后纠正两项旧计划记录：文本-only 图通过复制 CLIP 文本嵌入作为视觉占位，论文对此已经明确规定，不能继续列为未知；few-shot 的预训练书籍域是 Goodreads-NC，与常规评测的 Goodreads-LP 不同。

## 正式语料与任务

- 常规评测：ogbn-Products + Goodreads-LP + Amazon-Cloth。
- Few-shot：ogbn-Products + Goodreads-NC + Amazon-Cloth；5-way，1/3/5-shot，目标 Goodreads-NC、Arxiv、Wiki-CS。
- NC：Products、Ele-fashion、Goodreads-NC、Arxiv、Wiki-CS，冻结编码器后官方划分线性探测。
- LP：Goodreads-LP、Amazon-Cloth、Amazon-Sports，官方正/负样本 MRR 与 Hits；保留有向协议和反向重叠审计。
- KG：FB15K237、WN18RR，冻结节点编码器，用 head/tail 表征训练关系分类器，报告 accuracy。
- 主模型、CLIP-text/image 基线及论文的损失/融合/模态编码器消融需要独立配置与实验记录。

## 核验方法

CLIP 核验使用官方 raw text 与 node mapping，将候选 CLIP 重新编码后分别与特征的两个 512 维分块比较余弦相似度和误差。必须按原始 ID 对齐，不能只靠特征维度或预期准确率判断模态。一个数据集的证据不能自动充当另外四个数据集的独立证据。

OGB 数据保留官方 mapping 和 split；按原始节点 ID 对齐文本。不以 100 维官方数值特征代替论文的 CLIP 文本特征，不以子集冒充 2,449,029 节点完整 Products。

## 资源与验收

硬件仍为 RTX 3060 Laptop 6 GB / 约 16 GB RAM，E 盘本阶段开始时剩余约 37.8 GiB。模型维度 1024、三组各 6 个专家保持不变。预计算特征采用 CPU/磁盘存储，GPU 仅处理 CLIP 编码批量和 PPR 子图。大数据图和特征不能同时常驻 GPU，也不能不经预算全部载入 RAM。

验收：来源与版本固定、原始文件指纹明确、节点/文本 ID 对齐、CLIP 分块语义有可复核数值证据、联合采样覆盖与断点游标正确、实际 CUDA 前后向通过、评测使用验证选模且 test 不参与调参。

未披露的 PPR 参数、跨图混合比例、优化器/温度及 LP head 仍需查证或显式标记重建假设。不得通过把 stage 字符串改为 formal 来绕过这些要求，也不把梯度裁剪伪装为作者设置。
