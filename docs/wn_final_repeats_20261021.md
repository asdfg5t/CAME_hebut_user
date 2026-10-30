# WN18RR主任务06：Cloth之后的优先快速交付

2026-10-21，Asia/Shanghai。用户时间紧，要求接续一个准备最充分、较快完成的任务，训练由用户自行点击。入口为 `scripts/run_18_wn_final_repeats.py`。

## 选择依据及本阶段自审

Cloth已完成10/10，均值MRR×100=26.78319893367，样本标准差0.21622048325个百分点，比论文28.88低2.09680106633点。全部官方排名及产物哈希复核见 `artifacts/audit/cloth_ten_seeds_complete_verified_20261021.json`，不重训不覆盖。当前完整主任务重复完成7项，460个下游结果；不能称1471历史网格或整篇论文完成。

剩余主任务WN18RR、Goodreads-LP和Products中，WN只有40943个节点，完整1024维归一化编码缓存已存在，官方数据已齐全；只训练头尾拼接的2048→11线性关系分类头，远小于LP大规模候选评分或Products大图重新推理。此前同一固定方案seed0训练447轮（选347轮），从首次验证日志至完成约3分27秒。当前硬件/文件开销可能不同，暂按剩9个seed约1–3小时预留，不是保证时长；若较早停止通常更快。Products容量门槛尚未通过，Goodreads-LP协议可比性仍未解决，因此不优先派发这两项。

目标：使用已存在最高完整validation选定的WN方案，先验收再确认一次seed0，之后交给用户运行固定seed1..9，所有结果保留。科学依据为论文附录C：关系分类accuracy、冻结编码器、头尾拼接线性CE、官方全划分、10seed，不能错用LP的MRR。单位L2归一化、训练对统计仿射条件化/折回FP64线性评测、Adam及100patience属于明确工程适配，作者完整细节仍未确认；历史测试已被开发查看，旧至少+.5pp诊断拒绝结论不改，本次是独立最终交付阶段，不事后声称旧修复通过。

固定方案：原联合编码器SHA1fd5a614766cf52fe15cc9e0a9f577fbfdc07cd70de7bd6cc7a02834a1cde91e，完整40943×1024归一化缓存、train_reverse图政策保持。选用run20261018T015003Z_wn_unit_l2_c_353346_s0_203758，其完整3034验证准确率80.29004614370%，高于同池已完成候选；验证头已保存，不重训seed0。官方训练86835、验证3034、测试3134，11种关系；Adam lr.05、完整batch、2000轮上限、100patience，每seed只按validation选轮。测试分数不选方案、不挑seed，也不设2pp准入来阻止完成。

资源预算：单GPU串行；额外项目128MiB、外部缓存256MiB，保留每盘3GiB余量。无新增下载、无新预训练、无重编码全图、无降维/截官方测试/CPU神经计算回退、无删除原始数据/已完成Cloth或其他结果。

验收：相关15项CPU检查通过后冻结入口/配置/源头/缓存及官方manifest身份；实际Torch+DGL CUDA；相同Adam lr.05的3轮连续与2轮暂停后新run续至3轮对比头/Adam/全部RNG/采样generator/历史/最佳状态/下一轮逐项一致，小型32行fixture明确不算基准结果。完整归一化缓存SHA/维度/有限值/单位范数与官方全部划分资产核验，再用保存折回线性头完整重放3034验证，预测逐项完全一致。任何验收失败先保存错误并停止，不发布假通过。

## 用户只点击一个文件

在PyCharm运行 `scripts/run_18_wn_final_repeats.py`，参数留空。基础Python39自动转交项目`.venv-came39`。

默认：核验/复用准备结果→复用已保存seed0验证头及一次完整测试→固定seed1..9串行完整训练、验证选轮、各完整测试一次。运行中已完成任务按产物SHA复用；训练恢复完整Adam/RNG/历史，测试恢复已提交query游标，不重复已提交测试。新run记录父SHA是续训历史，不代表从头训练。低分不会丢弃或自动换策略，不能保证距论文81.97仅差2点。

代理只运行 `--prepare-only`：完成实际验收和一次完整seed0确认后状态为ready_for_user_repeats，正式seed1..9未由代理启动。用户默认点击后会核验复用，不重复seed0测试。`--dry-run`仅检查冻结身份；`--check-only`仅GPU验收；`--pause`请求完整轮/测试游标暂停，再次直接点击默认入口续跑，只归档清除WN自己的暂停，不改变其他队列暂停。

账本 `artifacts/workflow_state/wn_final_repeats_current.json`；日志/逐seed统计/图为 `artifacts/wn_final_repeats/`；配置 `configs/wn_final_repeats_20261021.json`；冻结清单 `artifacts/delivery/wn_final_repeats_release.json`。每run保留config/数据-环境-源码manifest/human logger/JSONL/CSV/TensorBoard/完整检查点/最佳头/训练曲线/所有官方测试pairs-labels-predictions。报告accuracy均值与样本标准差并注明n/10，一个预训练骨干上的十个头初始化不是十次独立预训练，更不是整篇复现完成。

## 实际验收与交付结果（2026-10-21上海）

准备已实际完成，账本为ready_for_user_repeats，记录仅runtime_check和test_s0，seed1..9未启动。真实Torch+DGL CUDA、完整Adam/RNG/采样/历史断点恢复、官方全部资产与3034验证逐项预测核验通过。相关15项检查和基础Python39启动通过。

seed0已选保存权重仅确认一次完整官方测试：2477/3134正确，accuracy=79.03637523931079%，比论文81.97%低2.933624760689213个百分点，尚未达到2pp目标；不依据该分数换方案。用户现在直接运行本入口，参数留空，会复用准备及seed0，继续其余9个seed。独立核验、CSV/JSONL/TensorBoard一致性、父SHA、恢复一致性及实际目视PNG证据见artifacts/audit/wn_final_preparation_verified_20261021.json。全部依赖本地，断网可以继续；仍需开机、接电。
