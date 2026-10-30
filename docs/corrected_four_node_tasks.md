# 四项节点分类的固定修正流程

当前范围为 Ele-fashion、Wiki-CS、Arxiv、Goodreads-NC。用户自行在 PyCharm 启动训练，本次交接不自动启动完整重复实验。根目录作者代码、论文、原始数据、旧实验保持不变；旧有向设置下的 Wiki-CS/Arxiv/Goodreads-NC 结果保留为历史记录，不能混入修正后的均值。

## 已确认的基准

Ele-fashion 保留十seed均值84.1524%，论文85.13%。修正后的seed=0结果：Wiki-CS官方split=0为77.6125%（论文78.57%），Arxiv70.9154%（论文71.66%），Goodreads-NC82.0964%（论文83.21%）。三项官方测试节点和标签与保存预测逐项核验通过。前三个修正后单seed结果不能代表完整重复实验均值。

冻结编码器SHA256为 `1fd5a614766cf52fe15cc9e0a9f577fbfdc07cd70de7bd6cc7a02834a1cde91e`：已有8个固定子集轮次、10240次更新，不追加预训练。Ele-fashion图保留原设置；另外三项采用先在验证集锁定的双向去重图，保持官方节点ID、标签及划分。论文模型宽度1024、完整专家及融合不变。冻结分类头为Adam/lr=.05/full-batch，上限2000轮、patience=100；Wiki-CS按官方stopping mask损失选优，其他两项按验证accuracy选优，然后测试一次。

作者完整图处理和部分优化细节未公开，双向图证据及选择锁见fast账本；本方案公开标注这些重建假设。随机初始化、CUDA计算及Wiki-CS官方划分可能导致分数变化，没有承诺最后均值与单seed完全相同，也不按测试分数挑seed。

## PyCharm顺序

用户最新要求只点一个训练文件。直接在PyCharm运行 **`scripts/run_00_main_pipeline.py`**，不填参数：它进入固定四项重复流程，自动复用Ele-fashion十次及其他三项已经确认的seed=0结果，随后运行全部剩余seed/split，保存每次结果、当前均值/标准差和四项对照图，完成后结束。它不会启动Products、三个LP任务、KG、few-shot或消融。`scripts/run_full_suite.py`是相同流程的兼容入口，不需要两个都运行。

本入口包含Wiki-CS 20划分×20seed，Arxiv/Goodreads-NC各10seed，Ele-fashion十seed复用，共430个结果；基准13个复用，起初417个待完成。`scripts/run_four_node_stability.py`保留为可选的分段诊断入口，不是默认流程的必需前置步骤。如果以前已运行这个可选检查，六个新增结果会核验复用，不重复训练/测试。

需要暂停时运行 `scripts/request_full_suite_pause.py`，等待 `SAFE_PAUSE` 以及 `CORRECTED_STATE=paused` 再关窗口。下次仍点击 **`scripts/run_00_main_pipeline.py`**，推理从缓存游标继续，分类器从含optimizer/RNG的检查点继续，新增run记录parent检查点。可选稳定性入口单独暂停时才用 `scripts/request_four_node_stability_pause.py`。

项目解释器是 `.venv-came39/Scripts/python.exe`；直接用用户默认Python39启动以上scripts时，入口会转交到项目环境。每个子任务实际检查Torch和DGL CUDA，不降维、不切CPU。

原1471作业历史模块保留，只有显式 `--legacy-full-suite` 才访问；来源改变时其保护会拒绝未经审查的旧账本续跑。原全节点主流程同样只有 `--legacy-main-pipeline` 才访问。默认点击不再自动进入旧方案。

## 预算、产物与验收

所有任务串行使用一张GPU，没有累计24小时硬截止。每个数据集共用一次编码缓存，批量Wiki/Arxiv=32、Goodreads=128，三组参数与本次确认一致。可再生quality缓存上限3GiB；整个数据集请求的seed全部完成后，只移除该模型/该数据集的完整派生缓存，原始数据、预测、日志及检查点保留。Goodreads约2.61GiB；如果此前确认后已清理，需要首次重编码一次，不能误解为重新预训练。

新指针为 `artifacts/workflow_state/corrected_node_stability_current.json` 与 `corrected_node_suite_current.json`。两者指向独立账本及结果目录。每个run继承人类logger、JSONL/CSV、TensorBoard、完整checkpoint、PNG与保存预测。聚合目录保存resolved_profile、源码/数据/环境/父实验指纹、plan、results.json、per_seed_results.csv、four_task_comparison.png；注明已完成/预计数量、暂定均值、样本标准差和论文差距。Wiki-CS跨seed/split的总体SD不冒称作者未公开的统计细节。

重复点击会检查所有已完成产物SHA256和完整官方测试成员；发生改变就拒绝静默复用。源码/配置运行中改变会停止派发下一任务。磁盘空间不足明确报错，不删除历史实验。单seed基准不会覆盖；重复实验平均值单独保存。四项统计完成也不代表论文十个主任务、few-shot和消融全部完成。

可用 `--dry-run` 核验计划而不训练，`--prepare-only` 保存计划及现有结果而不训练，`--single-seed` 仅复用已确认四项，不新增seed。`--max-new-jobs N` 是显式部分重复预算，不冒称整个实验完成。
