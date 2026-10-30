# 剩余四项主任务：单入口训练交付

## 只点这个文件

在PyCharm运行`scripts/run_14_remaining_tasks.py`。默认会通过现有`.venv-came39`执行，即使PyCharm仍选用户基础Python39也会自动转交，不修改全局环境。不要同时点击旧run08—run13入口。

范围为主任务06 WN18RR关系分类、08 Cloth链接预测、09 Goodreads-LP链接预测、10 Products节点分类，各固定seed0—9，共40个下游结果。之前六项450结果保留。这里的14是入口编号；40个结果不等于原1471个旧作业，也不包含CLIP基线、few-shot和消融。十个初始化共用已有一个联合预训练编码器，不冒称十次独立预训练。

当前run13诊断仍活动时，新入口只等待，不启动第二个GPU作业或打断它。诊断退出后，先验证交付SHA、原作者文件和实际Torch+DGL CUDA，再核验完整模型与直接CSC分块数值、评测游标与官方含平分排序的一致性。Products先完成独立原生DGL参考进程，再用新进程验证直接CSC完整邻居数值和512节点生产路径吞吐；参考进程的内存分配不会带入生产路径测速。通过后才允许全量续写，不改官方节点/边/测试或模型宽度。

## 暂停与继续

PyCharm停止/中断后，下次仍点同一个文件。入口复用已完成结果，训练从最后已提交完整轮的头/Adam或LBFGS/RNG/采样游标继续到新run，Products从原子提交编码游标继续。官方测试也保存完整固定头/数据身份、输出前缀SHA及逐批游标，恢复时不重算已提交查询。新run的manifest绑定父检查点SHA；历史run不覆盖。

更温和的暂停方式是在此入口的运行参数填`--pause`运行一次，然后移除运行参数，下次再点默认入口。该参数只创建本入口的`artifacts/workflow_state/remaining_tasks_delivery_pause.json`；不清除其他工作流暂停。新入口获得自身工作流锁后才归档自己的暂停标记并恢复；若已在运行，重复点击不会启动第二个GPU流程。

`--dry-run`只核对交付文件/配置，GPU任务0。`--preflight-only`只做实际GPU、精确采样与容量验收，正式训练0、官方测试0。

## 固定协议与已知限制

WN、Cloth只在声明过的、已完成的seed0完整验证候选中按validation最大值一次择优，平分保留先列者；未完成候选不参与。当前诊断的+0.5pp采用门槛和失败记录不改写；本文件是用户新授权的正式重复阶段，按标准验证择优冻结协议，不追认旧诊断达标，不根据test或论文距离挑模型/seed。选中后生成`protocol_selection.json`，绑定配置、摘要、最佳权重和缓存状态SHA；一旦开始正式结果，不能因新候选或低test重新选。

WN仍是1024维节点表示/头尾拼接线性CE，完整官方3134条测试；Cloth/Goodreads-LP仍是官方完整划分、训练边图、150固定负候选的MRR与反向重叠子集；Products保持2449029节点/1024 FP32/2213091官方测试。训练头/损失/归一化/优化器等未公开细节作为工程重建明确保留。Books只沿用已核验的固定协议，不改负例、不人为压低高于论文的分数。

当前可确认的历史test：WN77.9834% vs论文81.97%；Cloth26.8110 MRR×100 vs28.88；Goodreads-LP23.7908 vs10.42且论文协议可比性未确认；Products尚无完整test。已有单位L2或dropout的更高validation不等于新test。这个入口交付的是可核验训练流程，不能据此承诺四项最终都在2个百分点以内，不能称已实现论文同等效果。

所有seed均保留，包括较低结果，不依据结果提前丢弃seed。复用已有seed0验证权重，Books已核验完整seed0测试也复用；其他选定但未测试的seed0只用已冻结权重完整测试一次。seed1—9按同一固定协议训练与测试。只按validation选最佳轮，官方test不训练、不调参。原作者`model.py`、`fusion_modules.py`和PDF、原始数据及历史实验均不删除或改写。

## 日志与结果

- 总入口日志：`artifacts/remaining_tasks_delivery/pipeline.log`，UTC时间。
- 队列账本：`artifacts/workflow_state/remaining_tasks_delivery_current.json`。
- 逐seed结果：`artifacts/remaining_tasks_delivery/results.csv`与`results.json`，保留官方完整查询数、均值/样本标准差、论文差值和已完成seed数。
- 对比图：`artifacts/remaining_tasks_delivery/figures/remaining_main_comparison.png`。未满10次时标注实际n/10；Books协议未知单列。
- 每个真实run：`runs/<唯一目录>/`内的配置、源码/数据/环境指纹、人读logger、CSV/JSONL/TensorBoard和训练PNG；大断点位置由`checkpoints/location.json`指向项目既定外部缓存。暂停/恢复的日志分段保留并通过父SHA连接。
- 评测游标：该run对应检查点目录的`test_cursor.json`和`test_values.npy`；完整官方预测/排名文件保存在最终run。

额外预留项目512MiB/外部1GiB，不重新下载完整数据、不创建第二个Products全量嵌入。Products继续原124752行前缀，先校验SHA再写后续行。若资源/数值验收失败，明确保留Products为未完成并继续独立小任务，不能静默降维/切CPU/缩测试集。正式训练由用户启动，没有另加24小时硬截止；耗时以实测为准。

交付源文件由`artifacts/delivery/remaining_tasks_release.json`冻结，当前环境和每次实际运行仍记录到manifest。正式结果与全部PNG需在运行后按真实产物复核；代码验收不等于全部数值目标达标。
