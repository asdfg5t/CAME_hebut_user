# 剩余评测运行顺序

2026-10-21。已生成代码不代表已完成训练或达到论文成绩。当前八项主任务共有470份完整下游结果；Goodreads-LP重复由用户运行，Products尚待实际CUDA与速度验收。本文件只介绍新入口，不修改这些结果。

## 先运行Products

PyCharm打开 `scripts/run_24_products_sampled_stream.py`，参数留空，点击运行。无需先点另一个检测文件：同一入口等待Goodreads整个流程结束，实际检测Torch和DGL CUDA、数值、显存和吞吐；全部通过后继续十个独立线性头及完整官方测试。检测失败会保留原因并停止，不冒充正式完成。

两小时是有门槛的目标，目前没有真实吞吐或新准确率。每层10邻居是明确的推理近似，不能保证作者81.84%的准确率或2pp差距。`--prepare-only`可只检测；`--dry-run`只核对配置、不用GPU。`--pause`请求安全保存，之后同文件恢复。累计7200秒用尽会暂停，需要明确增加 `--extend-hours`，重启不会偷偷重置时间。

## 之后按需要点击一个独立文件

正式重复均参数留空，不需要先点旧1471入口。可以优先做25和26，得到论文的CLIP基线；27需要独立few-shot预训练；九消融耗时最长，可以按汇报需要选择先后。

- `scripts/run_25_clip_text_tasks.py`：CLIP文本，520项，包含Wiki-CS400次划分/初始化及30组few-shot配置。
- `scripts/run_26_clip_image_tasks.py`：CLIP图像，60项，只用于论文确有视觉的五个域。
- `scripts/run_27_few_shot_tasks.py`：论文few-shot语料的新预训练加30项评测，每项给1/3/5-shot结果。
- `scripts/run_28_ablation_reference.py`：四域、统一下游头的CAME匹配对照，40项；九个消融会自动准备并复用它，不必重复手动运行。
- `scripts/run_29_ablation_without_text_vision_loss.py`：去掉文本—图像损失。
- `scripts/run_30_ablation_without_text_fusion_loss.py`：去掉文本—融合损失。
- `scripts/run_31_ablation_without_vision_fusion_loss.py`：去掉图像—融合损失。
- `scripts/run_32_ablation_only_text_vision_loss.py`：仅保留文本—图像损失。
- `scripts/run_33_ablation_only_text_fusion_loss.py`：仅保留文本—融合损失。
- `scripts/run_34_ablation_only_vision_fusion_loss.py`：仅保留图像—融合损失。
- `scripts/run_35_ablation_without_moe.py`：用参数匹配MLP替换专家模块。
- `scripts/run_36_ablation_without_dimension_gate.py`：去掉逐维融合门。
- `scripts/run_37_ablation_without_modality_gnn.py`：先平均输入，再通过一个共享GNN，仅用于论文明确的这个消融。

九个消融各自41项，第一次还会完成共享对照；以后复用，不是每次都额外重做40项。新网格可执行位置总数1020，包括580个CLIP、31个CAME few-shot、369个消融及40个匹配对照。不是1020个epoch，不能直接从旧1471减去现有470冒充剩余完成数。

## 时间紧时怎么运行

上述25至37均可加 `--pilot`，只做每域seed0、Wiki仅split0；保持完整官方测试和完整声明的8轮预训练。它是诊断，不是正式十seed/400次统计；之后去掉该参数继续完整运行，会复用同协议已提交结果。

所有新入口支持 `--dry-run`、`--pause`。正常暂停或外部中断后运行同一个文件，保留已完成结果并从最新一致断点恢复到新run。神经训练检查点含优化器/RNG/游标，表征推理按已提交行恢复，LP测试按已提交查询恢复。Few-shot没有可训练头，支持任务级复用，未提交完整episode会重新计算该episode，不能声称支持episode内部的逐查询恢复。

新入口串行占用GPU，等待当前Goodreads/Products工作流释放。请一次只点击一个评测入口；重复点击会由锁拒绝。按预声明验证标准选模型，测试只作最终报告，不设置“分数低就自动更换策略”的循环。

## 耗时与交付边界

只有历史原主模型8轮/10240更新约70.65分钟是实测。新入口的CPU检查通过，不代表新GPU协议已跑通或计时完成；首次实际执行仍会检测CUDA、资源及数据身份。排期估计：CLIP两视图完整重复约8至24小时，完整CAME few-shot约3至8小时，一个消融约3至11小时，九个消融约1.5至4天；冷缓存、LP验证早停和机器负载可明显延长，不能保证截止时间。详细依据及未披露参数见 `docs/remaining_paper_tasks_20261021.md`。

结果在 `artifacts/remaining_paper_tasks/<组名>/`：state.json、results.json、measured_timing.json和pipeline.log；每项独立 `runs/` 保存配置/来源、human logger、CSV/JSONL、TensorBoard、PNG、官方预测或排名。检查点存于配置指定外部目录，并通过各run的checkpoints/location.json追溯；共享账本shared_jobs.json防止跨入口重复训练。

磁盘不足会停止并保留历史。只自动清理本批次命名空间、所有依赖结果已核验的可再生表征，不删除作者文件、原始数据、原有主模型缓存或已完成训练产物。仍有作者未公开的LP训练器、PPR/子集规模及few-shot距离等，均已单列，不能把工程重建称为作者逐项确认的完整复现。

本次只读空间检查：项目约33.29GiB/40GiB、外部缓存约27.41GiB/30GiB，仅约2.59GiB外部预算余量，实际磁盘尚有空间也不会擅自越过这个上限。参考模型的final214544714字节、完整last619077052字节；新模型Goodreads-NC完整FP32缓存约2.61GiB。新预训练入口启动前会将检查点保留量、最大活动表征与临时余量一并预检，当前预算很可能无法容纳一个新消融的峰值，更不能声称九个都可直接完成。需要先基于清单处理确认可再生的旧缓存，或由用户明确调整存储安排；本次没有删除旧缓存，也没有偷偷扩大上限。

启动前再优化：新入口已减少TensorBoard强制刷盘并复用有完整SHA/官方节点对/MRR证据的最佳头验证重放，不改变训练或测试集合。32项CPU检查通过，实际GPU提速尚未测量。复用八轮编码器只省去主模型对照的预训练；九个消融和正确语料的few-shot仍需独立预训练，不能把已有主模型换个标签当成消融。
