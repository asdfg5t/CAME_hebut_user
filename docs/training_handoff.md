# 用户自行训练手册

2026-10-11 协议修订：已确认论文附录 E 使用 Products 子集预训练和离线 PPR。下面全节点入口属于历史重建，不能当作论文必要成本。当前一天目标使用 PyCharm **CAME_Subset_Offline_24h** / `scripts/run_subset_day_pipeline.py`；细节见 `docs/paper_protocol_reread_20261011.md`，状态与验收见 `docs/current_status.md`。不会自动续接旧全节点配置；旧 run 保留。新子集精确恢复使用 `.venv-came39/Scripts/python.exe -m came_repro.subset_offline_train --config <该子 run 的 config.json> --resume <该子 run 的 last.pt>`，须保持同一固定池、模型、语料与配置。恢复新建 run，强制停止最多丢失最近一次原子保存后的步骤。

更新：2026-10-10，Asia/Shanghai。已完成数据/环境/入口验收；完整训练由用户启动。本手册区分论文披露设置与未披露部分的重建选择，不能据短程验收宣称达到论文性能。

## 1. 进入项目并预检

在 PowerShell 项目根目录执行。不要使用 PATH 的 python；无需重新下载数据或重新生成全部 CLIP 特征。

```powershell
Set-Location -LiteralPath 'E:\danzi\A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs'
$ProjectPython = '.\.venv-came39\Scripts\python.exe'
& $ProjectPython scripts/preflight_reproduction.py
```

预检核验作者原文件、十域完整特征与图文件指纹、特征归一化抽样、两套配置、实际 torch/DGL CUDA 和空间预算，输出 `PREPARATION_READY`。大文件校验可能需要数分钟。报告在 `artifacts/audit/preflight_reproduction.json`。正常现有环境不必重装；需要重建时才执行 `scripts/bootstrap.ps1`，它优先使用第二阶段完整依赖锁。

## 2. 启动完整联合预训练

两套语料必须分别训练。常规模型用于 NC、LP 和 KG；few-shot 模型用于 few-shot。

```powershell
.\scripts\start_joint_training.ps1 -Protocol representation -Seed 42
```

上述配置混合 Products、Goodreads-LP、Amazon-Cloth。另一套：

```powershell
.\scripts\start_joint_training.ps1 -Protocol few_shot -Seed 42
```

两者不要同时跑。默认维度 1024、三组各 6 专家、top-2、w12=.5/w13=.1/w23=.5、FP32、lr=1e-4、8 epochs、全部节点种子、无 max_steps 截断。未披露的 PPR alpha=.15/epsilon=1e-4、round-robin 混合、batch=32、全节点种子覆盖、Adam 默认参数和显式梯度裁剪 1.0 均记录在配置偏差清单。这是保留已披露参数的重建协议，不保证完全恢复作者未公开训练脚本。

脚本受 PowerShell 本机策略限制时，可直接调用同一解释器，无需修改全局策略：

```powershell
& $ProjectPython -u -m came_repro.joint_train --config configs/joint_representation_reconstruction.json --seed 42
```

只想自行重做有限验收：启动脚本添加 `-SmokeSteps 3`。此值是累计 optimizer 更新上限，结果标为 partial；不要把它用于正式结果。

每个 epoch exhaust 所有语料种子，约 10 万个 seed batch；8 轮约 80 万个 batch。无负样本 singleton 会明确记录跳过，batch 数不等于实际更新数。短程最高 torch allocated 约 2.64 GiB 仅适用于已测批次，长期预算可能多天，不据前三步外推精确完成时间。

## 3. 日志、检查点与恢复

控制台输出 `RUN_DIR=`；`runs/<run_id>/` 包括已解析配置、环境/代码/数据指纹、源码快照、UTC human logger、逐步 JSONL/CSV、TensorBoard、status/summary 和 PNG。北京时间为 UTC+8。

新大检查点在 `configs/storage.json` 的项目专属 C 盘目录；每个 run 的 `checkpoints/location.json` 和 summary 给出实际路径。联合训练在首步、每 250 次更新、epoch 边界和正常结束保存原子 `last.pt`，同一 run 不累积数百个大副本。完整 8 轮结束另外保存权重型 `encoder_final.pt`，其 `completed_epochs=8`；正式下游优先使用这个文件。

不要在运行中改配置/代码。中断或失败后读取 status/logger，从最近一致 last checkpoint 恢复；未保存的步骤会重新执行。恢复自动新建 run，记录父文件 SHA256，保存 optimizer、Python/NumPy/torch/CUDA RNG、每图随机顺序/游标和混合位置。DGL CUDA scatter 仍可能数值非确定，不承诺所有大图 bitwise 回放。

```powershell
$PretrainLast = 'C:\Users\DELL\came_reproduction_cache\01a12085-0d33-7121-8ca9-6a18d16a84f4\checkpoints\<实际run_id>\last.pt'
.\scripts\start_joint_training.ps1 -Protocol representation -Seed 42 -ResumeCheckpoint $PretrainLast
```

恢复需使用原 seed、同一语料与模型/采样配置。改变 batch/PPR/种子预算是新实验，不能接旧游标；程序会拒绝。改变后保存为新配置并记录适配，不自动降维、换 CPU 或以 BFS 代替 PPR。

TensorBoard：

```powershell
& $ProjectPython -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

访问 `http://127.0.0.1:6006`。损失对应不同大小 PPR 子图、不同负样本数，锯齿不能单独用于判断收敛。PCA/专家使用率为有范围声明的诊断，不是任务成绩。

## 4. 冻结模型后的评测

填入自己完整训练的权重路径，而非现有九步验收 checkpoint：

```powershell
$RepresentationEncoder = 'C:\Users\DELL\came_reproduction_cache\01a12085-0d33-7121-8ca9-6a18d16a84f4\checkpoints\<常规完整run_id>\encoder_final.pt'
$FewShotEncoder = 'C:\Users\DELL\came_reproduction_cache\01a12085-0d33-7121-8ca9-6a18d16a84f4\checkpoints\<few_shot完整run_id>\encoder_final.pt'
```

节点分类示例：

```powershell
& $ProjectPython -u -m came_repro.downstream --checkpoint $RepresentationEncoder --dataset ele-fashion --task node --seed 0
```

`--dataset` 可选 `ogbn-products`、`ele-fashion`、`books-nc`、`ogbn-arxiv`、`wiki-cs`。冻结完整编码器，用官方划分训练线性头，lr=.05、max_epochs=2000；Adam/patience=100 是重建选择。默认 full-batch，显存预算不足时明确失败；指定 `--probe-batch-size 4096` 是需要在比较中标注的 mini-batch 适配，不会静默启用。

Wiki-CS 要保留全部 20 个 split：

```powershell
& $ProjectPython -u -m came_repro.downstream --checkpoint $RepresentationEncoder --dataset wiki-cs --task node --seed 0 --split-id 0
```

分别运行 split-id 0..19。使用 stopping mask 的 loss 选轮次，validation mask 另作报告，test 仅在选择后评测。JSONL 的历史字段名 `val_loss/val_accuracy` 表示选模集合；具体集合看 `config.json/evaluation.selection_split`，新 PNG 明确标为 stopping/selection。其他 NC/KG 用 validation accuracy 选模。

关系分类：

```powershell
& $ProjectPython -u -m came_repro.downstream --checkpoint $RepresentationEncoder --dataset FB15K237 --task relation --seed 0
& $ProjectPython -u -m came_repro.downstream --checkpoint $RepresentationEncoder --dataset WN18RR --task relation --seed 0
```

训练图只含训练三元组，冻结 head/tail 拼接表征，指标是 relation accuracy，不是 KG ranking MRR。

链接预测：

```powershell
& $ProjectPython -u -m came_repro.link_evaluation --checkpoint $RepresentationEncoder --dataset sports-copurchase --seed 0
```

三个目标为 `sports-copurchase`、`cloth-copurchase`、`books-lp`。Hadamard 二层 MLP、Adam/lr=.001、100 epochs/patience=10、训练均匀非边负样本均为重建选择；验证/测试使用官方固定负候选，tie-aware MRR/Hits。验证 MRR 选模，test 一次。官方有向指标与排除反向重叠查询的子集分别报告；后者不是重新划分的官方 benchmark。严禁给 LP train 图盲目加逆边。

Few-shot：

```powershell
& $ProjectPython -u -m came_repro.downstream --checkpoint $FewShotEncoder --dataset books-nc --task few_shot --seed 0
```

另两个目标为 `ogbn-arxiv`、`wiki-cs`。每次同时保存 5-way 1/3/5-shot，明确记录采样类、support/query IDs、预测及 accuracy。余弦 prototype 和所选类别剩余官方有标签节点作 query 是重建选择，support/query 不重叠；不把 prototype 误称为作者未提供的分类器实现。

全图推理按所有 inbound 邻居、两层原 GAT、eval mode 分批，已做数值一致性验证，不以 PPR 子图推理替代。缓存绑定 encoder SHA256 与数据协议，可中断后接着生成。大图/高连接度节点仍可能 OOM；显式降低 `--inference-batch-size`（例如 8 或 1）只改变执行批量，但不能保证单个极密节点在 6 GB 内可推理。遇到单节点容量上限需单独处理，不自动牺牲邻居覆盖。

## 5. 下游恢复、基线与消融

分类头每 25 epochs（首轮/末轮/early stop 也保存）保存 `probe_last.pt`，LP 每轮保存 `link_head_last.pt`。状态含模型、optimizer、RNG、NumPy sampling generator、完成轮次/下一轮、best/stale、数据/配置、历史标量。只从 epoch 边界恢复；同一 epoch 内未保存的更新会重跑。

```powershell
& $ProjectPython -u -m came_repro.downstream --checkpoint $RepresentationEncoder --dataset ele-fashion --task node --seed 0 --resume-head 'C:\<实际路径>\probe_last.pt'
& $ProjectPython -u -m came_repro.link_evaluation --checkpoint $RepresentationEncoder --dataset sports-copurchase --seed 0 --resume-head 'C:\<实际路径>\link_head_last.pt'
```

保持原任务、seed、patience、batch/特征协议，epoch 总上限必须覆盖下一轮；已满足 early stop 的状态拒绝继续。head 恢复新建 run，保留父哈希；已完成有限测试的 run 只作工程验收，不依据它的 test 成绩决定训练预算。

CLIP-text 基线不用 encoder：

```powershell
& $ProjectPython -u -m came_repro.downstream --representation clip_text --dataset ele-fashion --task node --seed 0
& $ProjectPython -u -m came_repro.link_evaluation --representation clip_text --dataset sports-copurchase --seed 0
```

把 `clip_text` 改为 `clip_image` 可评测五个多模态域；文本-only 图不报告不存在的 image baseline。基线直接读取原有已核验特征，不制造另一份全图缓存。

消融配置在 `configs/ablations/`，用 `joint_train --config <配置文件>` 启动。包含三种去掉单项 loss、三种只保留单项 loss、去 dimension gate、参数近似匹配的真实 MLP 替代 MoE，以及均值输入后一个共享 GNN 的论文消融。最后一种只用于明确命名的消融，主模型始终保留两路独立编码器。消融选择保留原 surviving weights 的假设有说明；未执行完整消融结果。

## 6. 多 seed、汇总与空间

任务至少运行预先固定 seed 0..9，不挑 test 最好的 seed。Wiki-CS split 0..19 分开记录；如果每 split 做 10 seeds，就是 20×10 的本地完整网格。论文没有披露所有 seed 与 encoder 复用细节，报告时同时披露 encoder 哈希，不能把一个 encoder 的十次 probe 说成十次独立预训练。

把预先确定的十个同协议 run 路径传给汇总脚本：

```powershell
$SelectedRuns = @('runs/<seed0_run>','runs/<seed1_run>') # 填全十个，勿按 test 分数挑选
& $ProjectPython scripts/aggregate_results.py --runs $SelectedRuns --output 'artifacts/results/<新汇总名>'
```

默认要求 10 seeds/1 split、完整编码器和完整预设 head 上限，拒绝重复 seed、不同代码/数据/模型协议的混合。Wiki-CS 加 `--expected-splits 20`。保存每 split 的 mean/sample_std（ddof=1）与 seed 原值；split 与 seed 方差不混为一个数。有限验收仅在显式 `--allow-partial` 时可汇总，始终不标为论文复现完成。

空间检查：

```powershell
& $ProjectPython scripts/check_storage_budget.py
```

默认 E 项目 40 GiB 上限、C 专属 checkpoint+embedding 30 GiB 上限、各盘至少 3 GiB 余量；写大文件前预算失败会停止。Products FP32 1024 维融合嵌入约 9.34 GiB；多 checkpoint/seed 缓存不能全部无限保留。逐个 encoder 做完目标任务后，可以删除该 encoder 哈希目录下可重新生成的 embedding 缓存；先确认保存了分类头/预测/metrics/图，并只清理明确可再生缓存。不要删除 raw、原作者文件、最终 encoder、父检查点或历史日志。空间策略集中在 configs/storage.json，禁止改动脚本隐式绕过预算。

作者模型的 dropout 可产生零嵌入与巨大有限梯度。主训练显式监测并裁剪，原代码不改；该适配、Windows 软件栈、batch 和未披露协议均须出现在最终报告。所有标准完成之前，保持 paper_reproduction_complete=false。
