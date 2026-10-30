# 用户优先两项的有限诊断

目标是先完成Cloth、WN18RR单seed诊断和真实测试，正式重复留给用户；绝对差2个百分点是目标，不能保证。保护六项已完成正式重复、作者文件、原数据和历史结果。

Products实际Python进程均已退出，旧账本encoding是陈旧状态。保存游标124752，FP32缓存已提交前缀SHA重新核验一致，未删缓存/重启Products。退出原因未确定。新增旧监督pause文件注明本次用户优先级，防止在两项诊断期间误恢复Products；原ledger/profile/source不改。证据artifacts/audit/priority_cloth_wn_preflight_20261016.json。

依据：论文附录C冻结encoder、KG头尾拼接线性头/CE和validation选模；作者未给出LP头和完整优化参数。已有Cloth最后20轮train BCE由0.07644降至0.07098，lr一直0.001，validation在早停窗口震荡。WN原Adam最后100轮validation accuracy介于62.43%和79.17%，高学习率震荡是诊断假设，尚未证明是差距根因。所有新选择仅使用train/validation曲线，没有按test/paper差距挑seed、负例、划分或阈值。

固定各一个候选：WN保留当前验证最优图/特征/1024维、seed0新线性头，初始lr0.05、Adam、上限2000、100patience；仅加入基于validation CE的ReduceLROnPlateau，factor0.1/patience20/min_lr0.00005。Cloth从上一validation选第97轮的最佳头开始新精修阶段，明确重建新Adam而非伪称同一断点恢复，lr降为0.0001，最多80完整训练正边轮/20patience；两隐藏层LayerNorm预测头、K4/train与reverse非边过滤、batch4096、全部图/划分/150固定负例均保留。不额外预训练、不新编码、不同模态不预先平均。

保持原预声明门槛：完整validation至少比参考提高0.5pp，WN阈值80.3286090969%，Cloth阈值30.6874078058%。不因接近论文而降低阈值；未达门槛保留旧test、不新增test。通过后锁保存头/summary/cache SHA，直接完整官方test一次；独立重算全部官方预测或排名、日志CSV/JSONL/TensorBoard、头/父SHA及目视图，达到2pp且验收完整才准备正式重复入口。

资源：单GPU串行，先较小WN后Cloth，额外编码缓存0，头/日志/验收预留512MiB；项目40GiB、外缓存30GiB、最少余3GiB守卫通过。禁止CPU模型计算回退、降1024维、裁官方划分。六项针对性CPU检查已通过；下一阶段先实际Torch+DGL CUDA及两任务seed1完整权重/optimizer/scheduler/RNG/游标恢复一致性验收，fixture不是benchmark分数。

入口scripts/run_priority_cloth_wn_validation.py，独立ledger artifacts/workflow_state/priority_cloth_wn_current.json，协作暂停priority_cloth_wn_pause.json。源/profile在启动时冻结；恢复到新run并引用完整父checkpoint SHA，原历史不覆盖。此入口只做两项有限单seed诊断，不能当作用户正式十seed训练入口，不启动基线/few-shot/消融/旧1471队列，定时自动化保持暂停。结果有可能不改善；届时如实报告，不依据test连续搜索。

检查点工程更新：WN每25轮、Cloth每5轮，协作暂停/早停/最终轮无条件保存完整optimizer/scheduler/RNG/history。CSV/JSONL/TensorBoard仍每轮写入；外部强制退出时从最后已提交完整检查点在新run恢复，可能重放尚未提交的完整轮次，不能把仅有日志的轮次称为已提交断点。输入schema兼容与这项IO改动经归档审计、8项CPU检查后重新实际CUDA验收。
