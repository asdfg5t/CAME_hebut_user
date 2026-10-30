# Cloth 成对排序损失单候选诊断

用户于2026-10-17要求继续修复Cloth。上一轮扩大无标签池及三学习率候选的最佳完整validation为30.143639%，未超过旧最强30.253474%；旧确认test26.810958%，论文28.88%。这些历史test已经看过，新的开发不能称完全盲测。

## 目标、依据、未知项

本轮只改变一个固定候选的训练目标：在原冻结CAME1024维表示和原两隐藏LayerNorm预测头上，用mean softplus(s_negative-s_positive)替换逐点BCE。采样仍为训练正边及其反向已知边以外的4个均匀非边取当前最高分。BPR目标有[Rendle等原论文](https://proceedings.mlr.press/r7/rendle09a.html)依据，但当前max-of4负采样和无正则化Adam不是BPR论文训练算法的逐项复现，不冒称其原算法，更不是作者确认的CAME设置。

本地CAME论文第14页附录C确认冻结表示、完整官方划分和validation选择，但未给出LP完整头、损失和优化器；第16页附录E的Products子集/PPR只涉及预训练，不授权缩小下游测试。官方页面本次复查仍未提供完整CAME LP训练器：[PMLR](https://proceedings.mlr.press/v306/he26at.html)。本轮目标是检查排序目标是否比重建BCE更合适，而不是已证明BCE有代码错误。

train BCE下降而validation MRR平台是现有日志观察；它不能证明优化目标是差距根因。新的BPR损失也可能不改善。较强旧编码器只根据已有validation证据复用，SHA1fd5a614766cf52fe15cc9e0a9f577fbfdc07cd70de7bd6cc7a02834a1cde91e；六项正式重复结果不改变。不新增骨干训练，不选择其他seed。

## 冻结配置及验收

- configs/cloth_pairwise_20261017.json在新模型结果出现前声明；seed0、lr.001、batch4096、两隐藏1024维LayerNorm头、K4负采样、上限300个完整train-positive轮、patience20。patience20为明确的预算适配，不是作者参数。
- 125839全节点和旧完整缓存、官方全部训练正边、121323验证查询、97456测试查询及每条150固定负候选保持。图仅由训练边导出，反向重叠按已有审计单列，不能冒称所有评测对均与训练图无无向重叠。
- 实际Torch和DGL CUDA先验收；合成fixture连续3轮与第2轮安全暂停后新run恢复，完整头/Adam/全部RNG/采样生成器/历史/选轮/next_epoch逐项比较。fixture不报作任务指标。
- 全部validation先完成；至少比旧最强0.3025347366384903高0.005才锁定权重和summary/cache SHA，直接确认完整test一次。未通过就保留旧test，不降低门槛或新增test来碰分数。正式十seed不自动开始。
- 通过门槛后的test需独立核验官方ID/排名/MRR/Hits/反向子集、所有CSV/JSONL/TensorBoard与父SHA；PNG需实际查看才能标为视觉通过。失败候选也保留完整产物。

## 资源、文件、暂停与离线

单张RTX3060串行，最多8小时累计活动时间；达到预算安全暂停。新增空间预留项目128MiB、外部512MiB，仍遵守40/30GiB高水位和每盘3GiB余量；不删除旧产物。所有数据已在本地，HF/Transformers离线，不需网络。

scripts/run_09_cloth_pairwise_validation.py为独立入口，默认前台流程，--detach启动隐藏且脱离聊天工具job的Windows后台进程，进程级申请减少空闲睡眠；关机/重启/手动睡眠或合盖仍会中断。--pause请求完整轮边界保存，主动暂停后--resume仅归档本流程pause；恢复新run带完整父checkpoint SHA。其它任务pause原样保留。

账本artifacts/workflow_state/cloth_pairwise_current.json；控制器/控制台日志与outcome.json在artifacts/cloth_pairwise_20261017。原作者文件、原训练器、前六项450正式结果和全部历史结果保留。新模块src/came_repro/cloth_pairwise.py明确从旧独立训练函数派生，旧文件SHA验证不变。旧run08不会自动改派新策略，本入口是独立新协议。

目标尽量接近论文，但成功与2pp范围都需实测；本候选不能称作者确认设置或完整论文复现。
