# Cloth交给用户自行断点续训

2026-10-19，Asia/Shanghai。用户要求自己继续训练，不从头开始。本次只执行CPU断点/结果/源码与数据身份核验及默认Python39的`--dry-run`，没有代替用户启动GPU训练。

## 在PyCharm只运行这个文件

`scripts/run_17_cloth_final_repeats.py`

运行参数留空，直接点Run。不要使用`--dry-run`或`--check-only`，它们只检查不会继续正式训练；也不需要`--detach`。默认Python39会自动转交项目`.venv-came39`，在scripts工作目录的实际启动检查已通过。

## 已有进度会保留

- seed0/1/2已完成，共3/10，完整官方测试MRR×100分别26.82418986676、26.79297381871、26.28234558962。当前三结果均值26.63316975836，样本标准差0.30422328715个百分点，不是完整十seed统计。
- 当前seed3最后日志到了148轮，但完整原子断点只提交至147轮，next_epoch=148。再次运行会从148轮继续，只重算未提交的该轮，不会重做1至147轮。
- 保存状态含头、Adam、Python/NumPy/Torch/CUDA RNG、负采样generator、完整147行历史、最佳头/验证分数/选轮及stale；最佳验证选141轮、MRR=.30254298087185844。
- 路径由工作流账本和每run的checkpoints/location.json自动解析，不需要手填checkpoint。当前断点SHA61902d97e46fc2b69a470113471fd306831879525f642ba7252da8c9ee89b345。
- 续训会新建run目录、记录父checkpoint SHA；这是保留实验历史，不能把新目录误认为从头训练。
- 已完成seed0..2会核验后跳过。启动日志中的seed0复用提示不表示清空进度；之后应出现`JOB train_s3 parent=...link_head_last.pt`，首个新完整训练日志应为epoch148，随后完成seed3测试及seed4..9。

## 本次检查与实际状态

实际后台训练/控制器均已退出，原工作流ledger里的running_train_s3是陈旧状态，退出原因未确认；没有用running字样冒称仍在训练。GPU与工作流互斥均空闲，本入口自己的暂停文件不存在。旧诊断的维护暂停保持，不能为续训去清除其他流程的暂停。

完整checkpoint配置/数据身份/有限数值、原作者与全部冻结依赖/完整缓存SHA、已完成训练与测试产物哈希、三个完整官方测试source/target/rank/MRR均核验通过。审计 `artifacts/audit/cloth_self_training_handoff_20261019.json`。核验过程中未改模型/配置/实验协议/历史ledger，也未删除文件或重测测试集。

实时状态看 `artifacts/workflow_state/cloth_final_repeats_current.json`，人读进度看 `artifacts/cloth_final_repeats/pipeline.log`；汇总结果和图在同目录results.json、results.csv、cloth_final_comparison.png。当前原断点和已完成结果均留在原位置，程序会自动接续。运行期间不修改源码、配置、数据或移动缓存。

以后要安全暂停，可用同一文件带`--pause`请求保存完整轮/测试query游标；等待进程退出后，下次参数留空直接运行同一文件。强行停止时未提交轮需要重算。离线可继续训练，仍需保持开机与接电。
