# 单次质量试验与后续训练顺序

2026-10-12 用户最新要求：Ele-fashion 保留现有结果；其余先各完成一次实验，验证后再扩大；取消累计 24 小时截止，目标尽量接近论文，不承诺特定分数。

## 直接运行

先在 PyCharm 运行 `scripts/run_quality_pilot.py`，或选择 **CAME_Quality_Pilot**。它支持用户默认 Python39 自动转交项目 `.venv-came39`，不需要安装新环境。

本次旧 1471 队列已经安全暂停，已完成结果及 Goodreads-NC seed=6 第100轮检查点保留。新入口有独立账本，不会从头重跑旧实验。

“先一轮”按每个数据集先完成一个随机种子 seed=0 实验实施，不是把分类头只训练一个 epoch。分类头保留 lr=.05、最多2000轮、patience=100，正常按验证选择并早停。Wiki-CS 先用预先指定的 split=0；它不能代表全部20 split。旧 Ele-fashion 的模型及结果均不重训。

## 第一阶段：三个偏低的分类域

1. Wiki-CS、Arxiv、Goodreads-NC 各一个 CLIP-text 诊断；只训练/验证，不评测试集。
2. 复用上述三域已完成 seed=0 的原 CAME 验证成绩，无需重训旧模型。原测试已在历史实验评测过，此处不把它用于配置选择，也不伪称它从未评测。
3. 三域各一个补反向边/去重的图处理对照。Wiki-CS 有DGL加载器依据；Goodreads-NC 的MM-Graph GNN入口有该处理；作者CAME精确实现仍未公开，Arxiv候选也标为重建选择。所有新图只在内存构建，保留节点ID；不改原始数据，尤其不改LP训练拓扑。
4. 从完整旧训练检查点继续一个固定子集轮次：保留模型、Adam、RNG、固定池、采样游标；预计新增1280次更新，从10240到11520。它是8轮后的额外第9个子集轮次，不是论文原始8 epochs，也不是从头仅训1轮。Products/Goodreads-LP/Cloth 的固定池仍为16384/16384/8192，初次试验只增加训练，不同时改变数据量。
5. 新模型在上述三域各做一个验证诊断。预先固定的选择规则：平均验证准确率至少提升0.5个百分点，且任何域不退步超过1个百分点。图处理对照也须验证提升至少0.5个百分点才能选用。所有门槛与测试分数、论文测试目标无关。
6. 有稳定验证收益才冻结选择并做每域一个seed测试确认。没有收益则保留原模型、记录“需要继续核查”，不自动启动多seed或消融。只完成单seed不能认定论文统计复现成功。

## 第二阶段：先看报告，再跑其他任务

第一阶段结束后读取 `artifacts/workflow_state/quality_pilot_current.json` 指向的报告目录：

- `quality_results.json` / `.csv`：每个任务的验证/测试成绩及原始run路径；准确率和LP的MRR分开存放。
- `validation_comparison.png`：仅比较验证成绩，不拿验证准确率与论文测试准确率混比。
- `graph_selection.json`、`selection_lock.json`：图方案和模型选择，测试前冻结。
- `pipeline_*.log`：阶段日志、耗时、暂停/失败状态。

每个训练run继续保存完整配置、代码/数据/环境/父检查点哈希、logger、JSONL/CSV、TensorBoard、checkpoint和曲线。旧实验原样保留。

确认第一阶段值得继续后，在PyCharm运行 `scripts/run_first_seed_tasks.py` / **CAME_First_Seed_Tasks**：Products分类、Sports/Cloth/Goodreads-LP链接预测、FB15K237/WN18RR关系分类各只做seed=0。该入口默认不启动10seed、400次Wiki网格、few-shot或九种消融；这些仍是后续正式复现范围，需要质量审查后安排。

没有额外验证收益时，不声称“已经接近论文”。后续候选的数据量/混合方式可以调整，但应一次只改变一项、独立建run；不能反复看测试分数挑配置。

## 暂停、续跑与资源

运行中点击 `scripts/request_quality_pilot_pause.py` / **CAME_Quality_Safe_Pause**，等待 `SAFE_PAUSE` / `QUALITY_STATE=paused`。再运行同一入口会跳过已完成且哈希核验一致的任务，恢复未完成分类头、联合训练或图推理缓存；新run保留父检查点哈希。

新流程没有累计24小时或4小时截止。运行耗时仍写日志；Windows运行时钟排除系统睡眠。运行期间申请防止自动空闲睡眠，退出即释放；不能阻止手动休眠或电池保护休眠。

新诊断嵌入同时最多3GiB。每个已用完的自有派生缓存自动清理，保留原始资产、历史结果、日志、预测和模型；不会删除旧队列恢复所需的Goodreads-NC嵌入。Goodreads-NC双向图推理使用明确记录的batch=8，其余默认32；Products仍使用既有显式batch=4。模型宽度/专家/损失、真实PPR和CUDA不降低。

所有阶段失败即停止并保存可恢复状态。完整图首次推理仍可能较慢；无时间限制不等于速度或准确率有保证。
