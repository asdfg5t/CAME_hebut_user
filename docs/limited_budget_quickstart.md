# 一天／半天的同方法缩小预算复现

2026-10-10，Asia/Shanghai。用户已明确接受减少训练预算和部分重复实验，目标尽量在一天到两天内完成。最后一次澄清希望半天／一天，因此新增 12h、24h 两档；原完整配置保留。

## 现在的训练是什么

旧日志里的 step 已经是真实训练更新。每一步执行 PPR 采样、CUDA 前向与 InfoNCE、反向传播、optimizer.step。sample 是 CPU 采样耗时，train 是 GPU 更新耗时；不是先花几天完成采样才开始训练。缓存复用能减少后续采样成本，但无法把约 80 万个完整预算批次变成几千个。

## 用户手动切换

用户选择了“在 PyCharm 手动停止旧流程，再启动快速入口”。准备好后点旧运行窗口 Stop，确认旧流程结束，然后选择 **CAME_Budgeted_24h** → Run。

也可以直接右键 `scripts/run_budgeted_pipeline.py` → Run，默认即 24h；基础 Python39 会按已有启动桥转交到项目 .venv-came39。半天版本选择 **CAME_Budgeted_12h**。

新入口不会改旧配置、停止用户进程、删除原始数据或历史 run。它检查已有 GPU 入口锁，旧流程仍运行时会拒绝并行。联合训练在第一步、每 250 次更新和正常结束写一致 checkpoint；强制 Stop 只能保证复用最近已保存的 last.pt，未保存的更新可能需要重做。源代码/配置与数据协议匹配时，新入口自动复用最近一致检查点，记录真实父哈希，不按 test 分数挑选模型。

不要再点 run_00_main_pipeline.py 期待缩减预算；它仍是原完整流程。

## 保留与减少的内容

保留作者文件字节、完整十域 raw/特征、两套正确预训练语料、1024 宽度、两路独立 2 层 GAT、三组各 6 专家、top-2、.5/.1/.5 损失、lr=1e-4、FP32、真实 top-128 PPR、官方划分与 LP train-only 拓扑、CUDA。

24h 档：常规编码器最多累计 10,000 更新、few-shot 编码器最多累计 6,000 更新（包含已恢复步骤）；预训练阶段分别最多 5h/3h。下游一个 seed，Wiki-CS 只跑官方 split 0；分类头最多 300 epochs/patience=50，LP head 20 epochs/patience=5。尝试 13 个 CAME 任务 run，覆盖所有十域 NC/LP/KG 及三个 few-shot 目标；再跑对应 CLIP-text/image 基线。选择两种消融，各 1,000 更新并在 Ele-fashion 做 NC。这个顺序优先产出小中型任务，Products 完整 NC 推理排在主任务后部。

12h 档：常规／few-shot 最多累计 6,000／4,000 更新，预训练阶段最多 3h/2h；分类头 150 epochs/patience=30，LP 10 epochs/patience=3；消融各 500 更新。覆盖 Ele-fashion/Wiki-CS/Arxiv NC、Sports LP、FB15K237 relation、Wiki-CS/Arxiv few-shot 七个代表任务及对应基线；大型下游任务明确省略。两个预训练语料仍完整，不用子数据集冒充原始数据。

两档都不是八轮完整遍历。原配置中的 epochs=8 不被篡改；通过显式 max-steps 在达到更新预算时结束，保存真实 completed_epochs（可能为 0）和采样游标。不能把限制的步骤或种子预算改名为八个论文 epoch。结果标为“论文方法、缩小预算”，不是完整论文性能复现。指标可能低于、接近或高于论文，全部按实际测试预测计算，不预设分数。

每个任务仍使用 validation（Wiki-CS stopping mask）选择模型后一次 test。保留官方 LP negatives、tie-aware MRR/Hits、反向非重叠子集与原始 query ranks。仅减少重复 seed/split，不生成十 seed 的标准差；只运行两种消融，其余七种明确省略。

## 时间／容量边界与可恢复性

12h/24h 是包括预检在内的总墙钟上限，另有各阶段／任务上限。到点停止自己的子进程、保留最近已落盘 checkpoint/缓存，并记录 completed/failed/time_budget_reached/deferred。清理退出可能比上限多十几秒。它限制运行时长，不能保证全部任务在截止前成功；剩余任务没有指标就保持空白，不填造假的结果。

全邻居推理保持与原模型同一计算，不以 PPR 推理替代。显存 OOM 时只显式重试 target batch=1，保留日志；这不改变邻居覆盖、维度或精度。如果单节点仍超过 6 GB，记录失败并继续其他独立任务。原始数据、已保存表征／预测不修改。

E 项目 40 GiB、C 专属缓存 30 GiB、每盘余量 3 GiB 的保护继续执行。容量／数据／CUDA 预检不通过就停止。预算停止不是训练成功；有效模型不足时对应 CAME 任务标为 deferred，已有 CLIP 基线也不冒充 CAME。

训练日志每一步／每轮、原子 checkpoint、RNG/采样游标、TensorBoard、训练/资源/任务 PNG 仍由已经验收的核心模块产出。新父目录 `runs/<UTC>_same_method_limited_24h_<id>/` 保存：

- `logs/budget_pipeline.log`：总流程 human logger、所用子进程与超时记录。
- `config.json`、`manifest.json`、`source/`：固定预算、软件/代码/原文件指纹及快照。
- `summary.json`：每个尝试的原始子 summary、失败/省略原因、encoder 路径与哈希、实际耗时，paper_reproduction_complete=false。
- `results.csv`：真实 accuracy/MRR/Hits/1-3-5-shot 指标、子 run 路径与状态。失败／未运行项保持空白，不用 0 替代。
- `omitted_scope.json`：省略的 seed/split、消融、大型任务范围。

比较时同时报告训练更新数、实际覆盖、一个 seed/split、较少 probe epochs、原有 PPR/混合/裁剪/环境重建假设。保留完整流程用于以后加预算，不把两种实验混在同一张正式统计表里。
