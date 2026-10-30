# 2026-10-11 论文重读与一天策略

## 核查范围与结论

用户提供 PDF 共 19 页，已重新逐页读取；第 6、7、14、16 页另经整页渲染检查表格、正文和单位。逐页提取与 SHA256 位于 artifacts/audit/paper_reread_20261011/。本文件仅记录证据与重建决策，不代表完成正式数值复现。

第 16 页附录 E 明确写明 offline precomputed PPR subgraphs，以及对大规模 ogbn-Products 使用 subset pre-training。第 13 页算法允许节点 mini-batch，并没有定义每轮必须覆盖全部节点。第 14 页要求 8 epochs，不能据此推出 8 次完整 Products 全节点遍历。旧配置的全节点轮次是重建者选择，之前的 8–12 天估计不是论文必要成本。

Table 6 的单位是分钟：CAME 平均联合预训练每轮 2.13 min，Goodreads-NC 推理 6.9 min，显存 14.67 GB，参数 53.6 M。论文服务器有 8 张 RTX 5090 32 GB 和 125 GiB RAM；未明确该计时使用多少 GPU，也没有给完整下载、CLIP 编码、离线预处理到全部评测的总时间。不能直接换算为本地 RTX 3060 Laptop 6 GB 的总耗时。

## 保留的论文与作者代码设置

- 常规联合预训练三域 Products、Goodreads-LP、Amazon-Cloth，不用单域 Ele-fashion 预训练冒充迁移实验。
- 两路独立 GAT、2 层、1024 维、dropout .3；每视图 6 专家、总 18、top-2；TV/TF/VF 权重 .5/.1/.5；lr 1e-4；PPR top-128。
- 配置要求完整完成 8 个所声明固定子集轮次，超过时限则如实记录实际完成轮次。子集轮次不是全图 epoch。
- 官方 downstream train/validation/test 不缩减；编码器冻结，probe lr .05、最多 2000 轮，validation 选模型；训练子集不按任何标签挑选。
- LP 训练拓扑只用训练边；保留完整原始图与特征，不删除节点来改变官方下游划分。
- 作者原文件不变；真实 Torch/DGL CUDA，保留 logger/JSONL/CSV/TensorBoard/完整检查点与源码、数据、环境指纹。

## 论文没有公开的内容

作者没有给 Products 子集节点数、比例、选择方法、子集每轮是否重抽、seed batch size、跨域混合比例、PPR alpha/epsilon/方向与完整训练入口。因此无法确认自选子集等同作者子集。附录 E 对 Products 子集有直接支持，没有同样明确说明 Books/Cloth 也使用子集。Book/Cloth 限额必须单列为笔记本预算适配。

还需保留 optimizer/temperature、CLIP 精确生成流程、probe patience、LP head 等既有重建声明。Windows/cu118 栈、梯度裁剪 1.0 继续是工程偏差。Wiki-CS 原文为 20 个官方 split、20 次随机初始化；旧工作流每 split 10 seeds 不能声称完全一致。一天主方案暂不运行 Wiki-CS。

## 质量优先的一天方案

默认入口 scripts/run_subset_day_pipeline.py，配置 configs/subset_day_pipeline.json 与 configs/subset_offline_24h.json。与旧 configs/budgeted_12h.json / budgeted_24h.json 的在线 PPR、硬步数限制、缩短 probe 方案分开。

1. 预检真实 CUDA、完整数据指纹、原作者文件及磁盘预算，最多 30 分钟。
2. 从同一 pretraining seed=42 初始化主模型。无标签均匀抽固定 seed 池：Products 16,384、Goodreads-LP 16,384、Cloth 8,192。完整图拓扑保留；每 seed top-128 PPR 可能涉及池外邻居。这些数量不是作者公布值。
3. 全部所选 seed 的 PPR 邻居 ID 在 optimizer 更新前离线准备，并缓存跨轮、跨消融复用。训练期禁止新增未准备 seed 的 PPR 扩展；每步仍构造 PPR union 的 induced topology，不能冒称全部边块已离线构造。
4. batch=32，每子集轮次 1,280 个 batch，8 轮共计划 10,240 个 batch；主训练含离线准备预算最多 8 小时。预算到达时只保留已原子写入的检查点，未完成则报告 partial。
5. 主结果聚焦 Ele-fashion。CAME、CLIP-text、CLIP-image 各 3 个下游初始化 seed 0/1/2，保留最多 2000 probe epochs 与验证早停；每 job 最多 25 分钟。论文 CAME 85.13±.03%、text 83.53±.02%、image 80.26±.05%。我们的 3 个 probe seeds 共用一个预训练模型，不代表多次预训练稳定性。
6. 只有主模型完成 8 个子集轮次，才运行去 TV loss、去 dimension gate 两个消融；固定池、初始化 seed 和训练轮次一致，从头初始化，禁止仅主模型暖启动。每消融最多 4 小时；不足 8 轮不进入可比消融结果。完成的消融各做 3 个同样下游 seeds。
7. 总监控时限 24 小时，涵盖该入口子进程的预检、离线准备、训练和评测。启动顺序保障先取得主模型/基线结果，后做条件消融；到时未执行项目记 deferred。24 小时是停止预算，不是全部任务完成承诺。

此方案主动减少实验范围与种子重复，不降低主模型维度，不换目标函数、不换 BFS、不削减官方测试集。另一个 few-shot 预训练语料、九个其他 downstream 域、七个其他消融、论文十 seeds 全网格暂不执行；完整数据仍在本地。新增 PPR ID 存储约 40,960×128×4 = 20 MiB 上界量级（另加 SQLite/索引），跨轮复用，不会按 8 轮复制。训练检查点与嵌入仍受原有 C/E 预算限制。

## 耗时与成绩的验收

先用真实三域、小池、原模型、CUDA 做短程验收，记录离线准备和 GPU 更新耗时；少数步骤不能验证全池最慢节点、长程收敛和全部推理容量。需要用户实际长训练后用保存的 validation 曲线和原始 predictions 复算 test 结果，再与论文比较。

不能保证只差 1–3 个百分点。旧 84.064% 是单域 Ele-fashion、40 步、单 seed 工程 pilot，和论文三域联合迁移协议不同，不能把数值差 1.07 个百分点当成此新方案的性能保证。报告必须列出子集数量、实际轮次、训练 seed、probe seed、所有未完成任务和方法偏差，不能填写没有运行的指标。

2026-10-11 实测：正式 batch=32 的三域 12 步短程通过，各域 128 seeds，最大 allocated 2725.8 MiB；启动预热后的 GPU 更新大多 .2–.35 秒。这段 PPR 命中旧缓存，因此另用独立 RNG=20261011 每域 64 个无标签随机种子测未缓存成本：Products/Books-LP/Cloth 均值约 .019/.043/.036 秒，未缓存样本分别 64/64/59。按均值外推全固定池约 22 分钟，仅估计 PPR 计算部分，不包含完整数据核验、CSR 构建、训练和下游，也不覆盖长尾节点。样本原值在 `artifacts/audit/offline_ppr_timing_20261011.json`；12 步训练原值在新 calibration run。该证据支持一天计算预算有可行性，不能证明收敛或性能差距。
