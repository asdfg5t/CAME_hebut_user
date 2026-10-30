# Products修复后正式运行

2026-10-22上海。真实CUDA、数值、内存和速度校准已通过；正式训练和官方测试尚未由代理启动。只点击 `scripts/run_24_products_sampled_stream.py`，PyCharm参数留空；不要保留代理验收时使用的 `--prepare-only`。用户默认Python39会自动转入项目venv，不需重装环境。

入口会核验并复用已通过的校准，继续完整训练+验证特征、十独立线性头及全部2213091官方测试节点。默认保留全部196615训练节点、39323验证节点、1024维FP32融合表征与原联合预训练权重；不重新预训练骨干。固定每层15邻居/batch128为明确推理近似，不声称作者确认或逐项完全一致。

## 已实际验证

- Run：runs/20261022T035018Z_products_fas_218254_s0_64de5a。
- 实际Torch和DGL CUDA、作者小图前向、packed-head梯度与独立头、暂停序列化后Adam/参数一致、固定采样拓扑重放、FP32前向重放通过。
- 完整4096个均匀无标签节点，保守吞吐1208.8775687494558节点/秒；旧无缓存真实候选23.329425571864387，旧失败证据保留。新候选同时修改缓存及采样，不能将差异全部单独归因某一组件。
- 同一16点精确前缀的mean cosine0.9666882753372192，通过0.95工程门槛；这不是96.67%分类准确率，当前test_accuracy=null、test_evaluations=0。
- 实际峰值CUDA4735708160字节，约4.41GiB；加载后可用RAM2088185856字节，约1.94GiB。内存门槛仍1GiB，不足则保存暂停。
- 4GiB常用原始输入缓存，缓存2097152个完整512维FP32原始CLIP行；其他行仍逐值读取原数据，没有删节点。校准中GPU命中约98.98%，命中/未命中CUDA数据均逐位验证。
- 22项相关CPU回归、默认Python的Products和CLIP入口dry-run通过；新图已目视，CSV/JSONL及32批日志独立核对，TensorBoard事件文件存在。

## 时间与断点

4032.33770659406秒（约67.2分钟）是按本次真实吞吐、1.25安全系数和1500秒分类头/初始化预留估算的未来完整流程量级，尚不是实测完整训练耗时。建议预留1至2小时，机器负载、头早停轮数、读写和恢复可能延长。

历史校准全部累计保留688.4691541999346秒，原7200秒活动预算不重置，剩余约108.5分钟。超时会安全暂停，后续可显式 `--extend-hours 1` 增加一小时；新预算须由用户明确设置。`--pause`请求完整行/优化器更新边界暂停，再运行同一个文件继续；PyCharm强制停止可能回到最近已保存边界。

正常内存或时间暂停现在记录state=paused并正常退出，不再把TrainingPaused打印成未处理异常。真实运行失败仍须诊断，不能承诺任何环境条件下零报错。不要同时运行第二个GPU入口。

## 产物和范围

控制器：artifacts/workflow_state/products_sampled_stream_current.json；日志：artifacts/products_sampled_stream/pipeline.log；每阶段独立run下有人类logger、CSV/JSONL、TensorBoard、配置与代码/数据身份、外部检查点指针和图。正式结束保存十列完整官方预测、逐seed准确率/均值/样本标准差与论文差值；测试仅一次最终报告，不按test换策略或挑seed。

验收审计：artifacts/audit/products_cuda_cache_acceptance_verified_20261022.json；校准图：artifacts/products_sampled_stream/calibration_verified_20261022.png。冻结release SHA9941099212ebae3e48304db92adcb61a3ef9c913a2a6bb4c80efea7508079b2a。原作者文件、Goodreads和其他已完成结果未改未删，旧Products精确前缀及所有暂停/拒绝校准保留。

资源及工程验收通过不代表论文81.84%准确率已经达到，也不能保证差距在2pp以内；须等完整官方测试结果后如实比较。尚未做的新任务不得计为完成。
