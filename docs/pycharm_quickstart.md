# PyCharm 点击运行顺序

最新需求（2026-10-11）：全 CAME 主任务与重复实验使用 **CAME_Full_Suite_Resume** / `scripts/run_full_suite.py`；暂停时另运行 **CAME_Request_Safe_Pause**，等待保存退出，下次仍点同一总入口自动续跑。累计 24 小时用尽不会重置，不承诺所有任务一天完成。此入口的覆盖、未实现论文分析和恢复粒度见 `docs/full_suite_resume.md`。

2026-10-11 重要修订：下文旧 `CAME_00_Run_Main_Pipeline` 使用全节点覆盖的重建预算，成本不是论文强制要求。用户目前的一天目标请运行 **CAME_Subset_Offline_24h** 或右键 `scripts/run_subset_day_pipeline.py`；流程与边界见 `docs/paper_protocol_reread_20261011.md`。新入口先离线准备固定池的 PPR，然后 8 个子集轮次，再 Ele-fashion 三 seed 主结果与 CLIP 基线，最后两个条件消融。总时限 24 小时，超时项目如实记 partial/deferred，不保证全项目或论文成绩。

2026-10-10，Asia/Shanghai。入口已新增，作者 model.py、fusion_modules.py 和论文保持原字节。完整训练由用户启动。

## 只点一次的总入口

在 PyCharm 运行配置中选择 **CAME_00_Run_Main_Pipeline**，点击绿色 Run；也可右键 `scripts/run_00_main_pipeline.py` → Run。

自动顺序：预检 → 常规联合预训练（8 轮）→ NC/LP/KG 多 seed 评测 → few-shot 联合预训练（8 轮）→ few-shot 多 seed 评测。每一步调用已有阶段，沿用 logger/检查点/图片/空间限制，任一步失败即停止。总入口另存 logs/pipeline.log 和阶段状态；各阶段及子实验日志保留。

这是一键 CAME 主流程。CLIP 基线和消融是独立对照实验，仍使用 06/07 入口及 training_handoff.md，不在总入口中隐式启动。编号脚本可以直接由用户基础 Python39 启动，会明确显示并自动转交给下文指定的 .venv-came39，无需填写运行参数。总入口 dry-run 和失败即停行为已经验收，没有替用户启动完整训练。

## 一次性确认解释器

项目实验解释器：

```text
E:\danzi\A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs\.venv-came39\Scripts\python.exe
```

它由用户默认 C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe（3.9.13）创建，依赖安装在项目隔离环境中。PyCharm 原模块配置仍指向基础 Python39；不要据此推断已经选中项目环境。

已添加 `.run/CAME_*.run.xml` 共享配置，逐项写入项目 `.venv-came39` 的 SDK_HOME、项目工作目录与无缓冲 UTF-8 输出。打开项目后可从运行配置下拉菜单选择 CAME_01_Preflight。如果配置尚未显示，可关闭后重新打开项目，或右键 scripts 中的编号文件运行。

用户实际运行显示 PyCharm 仍使用基础 Python39。现已修正 scripts/run_*.py：在导入实验依赖之前，检查当前解释器；如果是基础 Python，打印 Launcher Python 和 Starting verified experiment Python，并保留参数、工作目录、stdout/stderr 及子进程退出码，自动启动既有项目 .venv-came39。不会向全局环境安装依赖、修改默认 Python 或换 CPU。若项目环境不存在则明确停止。直接运行内部 -m 模块仍要求使用项目环境。

也可在 Settings → Python Interpreter（部分版本显示 Project → Python Interpreter）中选择／添加现有本地解释器，指向上面的 python.exe，避免启动转交。PyCharm 运行配置与工作目录说明参见 [JetBrains 官方文档](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html)。

已在命令行用同一解释器验证脚本，包括从 scripts 目录启动；未实际操作 PyCharm GUI 点击按钮，不能把配置文件已生成等同于已经观察到 IDE 按钮运行。

## 按顺序点击 Run

1. `scripts/run_01_preflight.py`（CAME_01_Preflight）：核验十域完整数据/特征指纹、作者文件、空间、实际 torch/DGL CUDA 与完整模型反向。看到 PREPARATION_READY 与 STAGE_COMPLETE=preflight 后继续。哈希扫描可能数分钟。
2. `scripts/run_02_pretrain_representation.py`：Products + Goodreads-LP + Amazon-Cloth 联合预训练，完整 8 epochs。等待 STAGE_COMPLETE=pretrain_representation。
3. `scripts/run_03_evaluate_representation.py`：冻结上一步完整编码器，顺序运行 NC、LP、KG。默认十个 seeds；Wiki-CS 二十个官方 splits，每 split 十 seeds。共 290 个任务 run，编码器是同一个预训练 seed=42 的模型，不是十次独立预训练。
4. `scripts/run_04_pretrain_few_shot.py`：Products + Goodreads-NC + Amazon-Cloth 的另一套联合预训练，完整 8 epochs。与第二步语料不同，需要独立模型。
5. `scripts/run_05_evaluate_few_shot.py`：三域、十 seeds，共 30 个 run；每个 run 同时记录 5-way 1/3/5-shot。
6. `scripts/run_06_clip_baselines.py`：CLIP-text/image 的独立基线。默认共 380 个任务 run；文本-only 图没有 CLIP-image 基线，不制造不存在的图像模态。
7. `scripts/run_07_pretrain_ablation.py`：只运行 configs/pycharm_workflow.json 中明确指定的一种消融，默认 without_text_vision_loss。其他九种配置的选择、后续冻结评测和汇总仍按 training_handoff.md 执行；不把一次消融说成九种都完成。

模型/张量计算在 CUDA GPU；CPU 负责图存储、PPR 采样、数据读取和日志。GPU 不可用时失败，不切 CPU，不改主模型宽度，不用 BFS 替代 PPR。预检、训练和批量评测的 PyCharm 入口有共享 OS 锁，一次只允许一个此类入口。TensorBoard 可以同时运行。

这些均为完整预算入口，没有 `--max-steps`。预训练 8 epochs 来自 PDF 第 14 页附录 C / Table 4。冻结分类头的最大 2000 epochs、lr=.05 与 validation early stopping 是另一阶段；实际轮数可能因 early stop 少于 2000。LP head 的 100 epochs 是未披露预测头协议的重建配置，不伪称论文原参数。PPR alpha/epsilon、全节点种子预算、混合、batch=32、Windows 软件栈、梯度裁剪等已在原配置声明差异。

## 直接运行时自动处理的事情

- 入口依据脚本位置定位项目根目录；不会因右键运行时工作目录为 scripts 而错误读取数据。
- 下游默认选择正确语料、pretraining_seed=42、8 轮完成、主模型 variant=came 的最近完成 encoder_final.pt。选择依据是完成时间，不是 test 成绩；已有有限步 pilot 会被拒绝。可在 configs/pycharm_workflow.json 明确填 representation_encoder/few_shot_encoder 固定路径。
- 多 seed 子实验依次运行。每个子实验独立保留现有全套配置/指纹、logger、JSONL/CSV、TensorBoard、checkpoint、任务预测和 PNG；工作流另外保存 plan/config/manifest、logs/workflow.log、状态与成功子 run 索引。
- 重复运行同一批量阶段，只跳过本阶段同一代码/配置/数据 manifest/encoder 指纹下已成功且摘要未变的子实验。不按 test 分数挑选历史结果。未完成子实验会新建 run 从头重做；想恢复头内部状态时使用 training_handoff.md 的 --resume-head，不宣称已恢复未保存游标。
- 原子检查点／全图缓存仍服从 E 项目 40 GiB、C 专属缓存 30 GiB、每盘 3 GiB 余量。多 encoder/消融缓存超过预算会明确停止，按手册只清理指定可再生嵌入，不自动删除原始数据、最终模型或历史日志。

预训练中断后，把实际 last.pt 路径填入 configs/pycharm_workflow.json 对应 representation_resume_checkpoint 或 few_shot_resume_checkpoint，再运行同一编号，保持原 seed/模型/PPR/预算设置。恢复新建实验目录并记录 parent SHA256。PyCharm 的强制 Stop 不能承诺保存正在进行的一步；从最近已保存的一致 checkpoint 继续。

## Logger 与可视化在哪里

每个实验目录 `runs/<run_id>/`：

- `logs/train.log`：human logger、进度、工程差异、异常；UTC 时间，显示时北京时间加 8 小时。
- `logs/metrics.jsonl`、`metrics.csv`：逐步／逐轮原值、训练 loss、选模指标、梯度/零嵌入、学习率、显存、耗时和覆盖范围（按对应任务记录）。
- `tensorboard/`：event 文件，及时 flush。运行 `scripts/run_tensorboard.py`，浏览器访问 http://127.0.0.1:6006。
- `figures/`：预训练 loss/资源曲线；正常结束的训练种子 PCA/专家路由诊断；probe loss/选模曲线；LP loss/validation MRR；few-shot accuracy。图标明来源、seed 和样本范围，原始数值另存。预训练曲线按 checkpoint 周期刷新，诊断图在正常结束生成，不承诺每一步都生成全部图片。
- `config.json`、`manifest.json`、`source/`：实际参数、代码／环境／数据版本和偏差。
- `status.json`、`summary.json`：阶段、成功/中断、实际预算与未完成复现标记。
- `checkpoints/location.json`：新大检查点的 C 盘真实位置；保存模型、optimizer、RNG、采样 order/cursor 或 head epoch 边界。
- NC/KG 保存预测/标签/索引；LP 保存 query ranks 与反向排除索引，能复算 MRR/Hits；few-shot 保存 support/query/prediction，能复算 accuracy。

第一步之后现在应运行第二步，不需重新下载或重新编码全部数据。6 GB 显存的短程与恢复已验证；长训练和大图所有密集节点仍有未覆盖容量边界，不能承诺绝对无错或与作者未公开设置完全相同。
