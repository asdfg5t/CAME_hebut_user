# CAME 学术复现计划与验收规范

最新单入口约定：用户只点击 `scripts/run_00_main_pipeline.py` 即完成固定四项重复流程，复用已完成结果，其他任务暂不派发。稳定性分段入口为可选工具，不需要先后点击两个训练文件。具体操作以 `docs/corrected_four_node_tasks.md` 和 `docs/current_status.md` 最新段落为准。

2026-10-13最新用户优先级：先完成四项节点分类，Ele-fashion十seed不变，另外三项按验证锁定修正协议重复；完整训练由用户自行点击。当前默认入口 `scripts/run_full_suite.py` 和 `scripts/run_00_main_pipeline.py` 均已转向固定四项流程，旧1471作业方案需显式历史参数，不再作为默认。先用 `scripts/run_four_node_stability.py` 补两个seed，再接续完整430结果网格。没有额外预训练或24小时硬截止；协议、资源、验收和已确认分数见 `docs/corrected_four_node_tasks.md`。下文日期更早的默认入口与时间预算是历史规划，以此段及最新状态为准。

制定日期：2026-10-09（Asia/Shanghai）。状态以实际实验产物为准，计划不等于完成。

2026-10-11 修订：完整重读 19 页后确认附录 E 的 Products 子集预训练与离线 PPR。旧全节点轮次预算不是论文要求，默认交接改为 `scripts/run_subset_day_pipeline.py` 的 24 小时质量优先方案；详细证据、未知参数、额外适配和验收见 `docs/paper_protocol_reread_20261011.md`。旧入口保留作历史重建配置，不作为论文必要预算。

同日最新范围更新：用户要求全部任务/重复实验和安全暂停续跑。新增 `scripts/run_full_suite.py` / **CAME_Full_Suite_Resume**，排齐全部 CAME 主任务、CLIP 基线、九个消融的完整下游 seed/split 网格，累计 24 小时预算不会因暂停续跑重置。该网格不包含其他算法重新训练或全部附录分析，且作者独立预训练 seed 细节仍未知；不能称为整篇论文全实验已实现，更不能保证一天完成。当前操作以 `docs/full_suite_resume.md` 为准；原重点入口保留。

## 1. 每阶段开始前的决策规则

依次核对：目标是什么、论文证据在哪里、现有代码是否一致、数据是否可追溯、硬件是否可承受、如何证明做对了。证据不足的参数必须进入偏差清单；不能把推测写成作者设置。阶段失败先定位原因，再更新配置与记录，不覆盖已有实验。

## 2. 已核实的论文设置

依据用户提供论文第 14 页附录 C、Table 4；该页已经渲染核对。

- 两路独立 GAT，各 2 层，隐藏维度 1024，dropout 0.3。
- 文本、图像、融合三组专家，每组 6 个，组内 top-2。作者 `MMgraph.num_experts` 表示总数，因此必须传入 18，不能直接传入 6。
- 三对损失权重：text-image 0.5、text-fusion 0.1、image-fusion 0.5；对应作者参数 `w12=0.5, w13=0.1, w23=0.5`。
- PPR 每个种子 top-128；预训练 8 epochs，学习率 1e-4。
- 第 16 页附录 E 明确离线 PPR 和 Products subset pre-training，未规定其子集大小与 epoch 节点预算。8 epochs 不能推出全节点覆盖 8 次。
- 常规评测联合预训练：ogbn-Products、Goodreads-LP、Amazon-Cloth；few-shot 联合预训练：ogbn-Products、Goodreads-NC、Amazon-Cloth（§5.2 第 6 页）。
- 文本-only 图复制 CLIP 文本嵌入作为视觉占位（§5.2 已明确），不增加其他模态。
- 冻结编码器后 linear probe：lr 0.05，上限 2000 epochs，validation early stopping；多模态任务 10 seeds。
- KG 任务是关系分类 accuracy，不能用链接预测 MRR 代替。
- 论文原环境：Linux / Python 3.9.23 / PyTorch 2.8.0 CUDA 12.8 / DGL 1.1.3；8 张 RTX 5090，每张 32 GB；Table 6 报告 CAME 显存 14.67 GB。
- Table 6 平均联合预训练每轮 2.13 分钟、Goodreads-NC 推理 6.9 分钟；不能把服务器测量值当作本地全流程耗时。Wiki-CS 为 20 官方 split、20 次随机初始化；旧每 split 10 seeds 的工作流不是该原文设置。

论文未充分规定的项目：CLIP 模型及预处理精确版本、拼接特征模态顺序、PPR alpha/epsilon/方向、种子批量、训练种子总数与跨数据集混合比例、优化器完整参数、temperature、预训练验证策略、完整 LP 预测头与训练细节。这些必须单独标记，不从结果反推设置。

## 3. 原文件保护与代码差异

根目录 `model.py`、`fusion_modules.py` 和论文保持原名、原字节；`artifacts/audit/original_files.json` 保存初始 SHA256。新增代码只写 `src/came_repro/`、`scripts/` 等独立目录。

作者模型引用缺失的 `.unigraph2.MoE`，但没有使用该符号。加载器只在内存中去除这一条 import，保留修改前后源码哈希与差异说明，不提供虚假 MoE 实现。其余作者模型直接执行。

实际较大批量验证揭示数值问题：attention dropout 可能删除只有自环节点的唯一消息，产生零嵌入，InfoNCE 归一化后首步梯度可达 6e13。新增训练层只添加监测和显式优化器梯度保护；`max_grad_norm=None` 时异常大梯度拒绝更新，本地工程配置显式裁剪至 1.0。原代码和目标函数不变，裁剪必须标记为未披露于论文的工程偏差，不宣称与原训练更新等价。

注意作者实现中的限制：`use_moe`/`fusion_type` 未真正控制融合路径；`use_gate=False` 会导致路由输入维度不一致；hidden/out 不等时也可能失败；有未被 forward 使用的子模块；分块 InfoNCE 对不等大小块取均值会改变节点权重，而且保存各块计算图，不能保证反向阶段显存线性降低。主模型使用 gate=True、hidden=out、作者完整 InfoNCE。新增 `variants.py` 为论文消融提供独立实例适配与有效性测试，不使用失效开关，不改作者文件；完整消融数值实验尚未运行。

## 4. 本地环境与硬件路径

用户指定默认解释器：`C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe`，Python 3.9.13。它已经在允许访问的工具环境下验证可启动；初次 sandbox 启动失败不代表解释器损坏。

项目环境：`.venv-came39`，基于上述解释器创建。默认解释器与原有 `.venv` 保留。未使用 `.venv-came`（3.11）已按用户空间要求清理，见 `artifacts/audit/unused_runtime_cleanup.json`。

Windows 首选兼容路线：PyTorch 2.1.2+cu118、DGL 1.1.2+cu118。这与论文软件栈有偏差，必须保留。DGL 官方 wheel 列表包含对应 cp39 win_amd64 包，不能只安装 CPU 版 DGL。`nvidia-smi` 的 CUDA 13.3 是驱动支持能力，不要求安装同版本 PyTorch。

检测 GPU：RTX 3060 Laptop、6144 MiB。先维持论文模型维度与专家数，只减小每步种子数并使用 CPU 存图/特征、GPU 存子图。该批量与硬件差异会写入 manifest。若 1024 维模型仍 OOM，单独建立降维工程配置并清楚标为非论文配置，禁止静默切 CPU 或自动降维。

## 5. 目录和命名规则

`src/came_repro/`：可安装 Python 包；模块和函数 snake_case，类 PascalCase，常量 UPPER_SNAKE_CASE。禁止在训练入口硬编码机器路径。

`configs/`：JSON 配置，英文 snake_case；实验名描述阶段和数据，例 `pilot_ele_fashion_cuda.json`。论文超参数与工程参数分层。配置写入每次 run，启动后不可修改。

`data/raw/mm_graph/`：官方文件原样存放，保留上游文件名；`manifest.json` 保存固定数据提交、下载大小和 SHA256。`data/processed/`：可重建的 PPR 缓存与特征变换；缓存键绑定原文件哈希、采样算法版本与参数。

`third_party/mm_graph/`：官方参考加载器/评测代码与许可证，不混入作者模型；来源版本和获取状态写 provenance。

`runs/<时间>_<实验名>_seed<种子>_<随机后缀>/`：禁止覆盖。子目录包括 `logs/`、`checkpoints/`、`figures/`、`tensorboard/`、`source/`。run ID 使用 UTC 时间，记录和报告显式注明 UTC 与 Asia/Shanghai。

`docs/`：计划、论文差异、环境说明、阶段结果；`tests/`：学术有效性相关的关键验证，避免只验证代码是否能导入。

## 6. 阶段与通过条件

### A. 环境和数据

真实 CUDA 张量运算、DGL CUDA 图运算、作者完整模型反向与 optimizer step 都通过。HF LFS 资产与发布者 SHA256 一致；无公开发布者校验值的来源记录官方来源和本地指纹，不能伪称完成发布者哈希比较。检查节点数、边 ID 范围、官方 split 不重叠、标签缺失处理及两模态维度与顺序。若特征语义尚未确定，只允许带明示假设的工程 pilot。

### B. 单数据集工程验证

Ele-fashion 真实数据，1024 维、18 总专家、论文损失权重、top-128 PPR，有限步 CUDA 训练。必须产出完整配置、环境/数据/源码指纹、每步三对 loss 与总 loss、梯度范数、学习率、时间和显存、TensorBoard、checkpoint、训练曲线、表征诊断、路由分布。缺失数据或非有限损失必须失败并保存异常记录。

验证 checkpoint 中 model、optimizer、Python/NumPy/torch/CUDA RNG、采样顺序及游标，恢复后从下一步继续。用小型实验比较连续运行与分段恢复的下一步行为；CUDA 算子存在非确定性时记录容差与具体限制，不能承诺 bitwise 相同。

### C. 正式训练和评测

正式配置不得使用 `max_steps` 冒充完整 epoch。完整预训练应混合 Products、Goodreads-LP、Amazon-Cloth；LP 预训练/推理图只使用训练边，不得使用验证或测试正边。NC 采用论文的无标签图表示协议，是否传导式使用全图必须明示。

完成官方节点线性探测、LP MRR/Hits、few-shot、KG 关系分类，测试集只在验证选模型后评测，不能据 test 调参。完整图嵌入抽取策略需要单独验证等价性；独立 PPR 子图推理与全图 GAT 不等价。

官方 Amazon LP 数据划分存在反向重复：Sports 的 valid/test 与训练图按无向节点对分别重叠 8,264 / 6,621 对；Cloth 分别 21,043 / 17,360 对，同方向重叠均为 0。详见 `artifacts/audit/lp_split_overlap.json`。因此不能盲目 `to_bidirected` 后宣称没有 test 拓扑泄露。正式评测需保留可比较的官方有向协议，并独立报告排除反向重叠的严格协议；生成派生视图时记录移除节点对清单与哈希，不修改官方原始文件。Goodreads-LP 也须做相同审计。

Goodreads-LP 实际审计结果：valid/test 按无向对重复 36,351 / 108,818，同方向重复 0。Goodreads-NC 的官方边数比论文 Table 5 少 36 条，且 7,372 个节点不在官方标签划分中；记录这些数据版本/计数差异，不伪造边，也不改变官方索引。

### D. 论文结果复核

论文完整评测为其他域 10 seeds、Wiki-CS 20 官方 split 与 20 次随机初始化；保存原始结果、均值、样本标准差、配置、预算与失败 run。一天适配优先 Ele-fashion 三个下游初始化、一个联合预训练 seed，不能称为论文全部统计复现。先做完整模型与 CLIP 基线，再做同池同轮数的真实消融；不报告未运行结果。

## 7. 数据下载范围与容量

官方来源：https://github.com/mm-graph-benchmark/mm-graph-benchmark ，其链接指向 https://huggingface.co/datasets/mm-graph-org/mm-graph 。

第一批下载五个多模态数据集的 CLIP 特征、图结构、划分、node_mapping，约 9.93 GB；不将 T5+ViT 特征当作论文 CLIP 特征。压缩原图合计超过当前 E 盘剩余空间预算，而且现成 CLIP 特征仍需确定原始两路顺序和 backbone 才能用于正式比较。

第一批未覆盖的 ogbn-Products、ogbn-Arxiv、Wiki-CS、FB15K237、WN18RR 在 2026-10-10 已补齐完整图、文本与 CLIP 特征。没有以 MM-CoDEx 替代 KG。数据来源/固定版本见各 manifest，当前容量与验收见 `docs/current_status.md`、`artifacts/audit/preflight_reproduction.json`。

## 8. 日志、恢复与学术记录

`train.log` 保存人读日志和异常堆栈，`metrics.jsonl` 与 `metrics.csv` 保存同一份逐步数值且及时 flush，TensorBoard 保存标量。不得用 test 选择预训练最佳 checkpoint；只保存 last 与规定间隔 checkpoint，正式 best 由单独规定的 validation 协议确定。

训练产出 `config.json`、`manifest.json`、`status.json`、`summary.json`、last checkpoint、loss/资源曲线与诊断图。新大文件按 `configs/storage.json` 写入项目专属 C 盘缓存，run 的 `checkpoints/location.json` 指向实际文件；不会每次保留一个完整 optimizer 历史副本。checkpoint 临时副本计入预算，原子替换；恢复新建 run 并记录 parent 路径/哈希。

所有图注明数据来源、样本范围、seed 和变换；PCA/t-SNE 只是诊断，不能作为指标替代；诊断节点不含测试标签。专家使用率与门控权重保留数值文件，图片只是其可视化。

## 9. 参考资料

- 本地用户论文，附录 C/D/E 和 Tables 4/5/6。
- DGL CUDA Windows wheels：https://data.dgl.ai/wheels/cu118/repo.html
- PyTorch 官方历史版本：https://pytorch.org/get-started/previous-versions/
- 官方 MM-Graph 数据与加载器，链接见第 7 节。
