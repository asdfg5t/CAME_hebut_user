# Goodreads-LP：官方依据重建与真实验收

2026-10-21，Asia/Shanghai。用户最新范围：只优先Goodreads，先完整单seed实测，再由用户完成重复；撤销2pp准入，以尽量对齐论文协议和如实报告为目标。不能用测试分数调参或刻意降分。

## 本阶段自审

目标：以固定官方基准的LP实现补齐CAME未公开的头训练细节，交付单入口、真实CUDA、标准logger及一致断点。论文附录C第14页明确官方划分、验证选模型及10seed；没有披露Goodreads-LP完整预测头/负采样器。查阅PMLR论文页https://proceedings.mlr.press/v306/he26at.html，页面只列PDF及OpenReview，没有新增可核验训练器。因此以下是有来源的重建，不能称作者确认设置。

证据：本地只读固定MM-Graph archive提交0769fa670c69a6dc0756a2d7fc697b980e494775，archive SHA491a0056d7767063def646052881852159fd071e6966b1533361083642c9cd96。lp/models.py:20-25的SAGE predictor为Hadamard后1024→1024→1024→1、ReLU两次（官方hidden_dim默认256，保留本项目1024是显式适配）。lp/main.py:86,121-139在batch512时每优化轮最多1000步，即512000正边；不得称完整数据epoch。lp/main.py:100-101为Uniform(1)训练负采样，127-131拼接正负分数BCE mean；configs/defaults.yaml为Adam lr5e-4、StepLR每5轮gamma.1、30轮。CAME要求验证early stopping，此处patience10仍为重建假设。冻结原1024维编码器及完整636502行缓存，保持原train-only有向图；官方基准的GNN联合训练和双向图不能悄悄取代CAME冻结表征。

未知：CAME真实LP头/优化器/训练预算/图方向细节仍未确认。采用此方案是对齐可验证基准，而不是证明旧23.790805%错误；旧结果保留。新的方案不以接近论文10.42的程度选择，不过滤/重采官方候选，不缩测试集。每轮从完整官方训练池无放回抽取512000正边，轮间重新打乱，train Uniform仅使用节点数；不检查val/test去排除负例。完整570464验证及1710677测试，每查询固定150负候选，含发布数组中的自身正target等原值全部保留。反向训练重叠108818个测试查询单列，不能把排除后的子集替代官方MRR。

资源：一个GPU串行，复用现有2.428GiB冻结编码CUDA行缓存，不重编码、不下载、不追加预训练；预计额外项目256MiB/外部缓存1GiB并保持磁盘3GiB余量。seed0最多30优化轮，每轮完整验证，实际耗时须实测，不能使用已删除模拟入口的“1小时”。训练中间检查点保存当前轮训练顺序/已处理batch/累计loss、头/Adam/StepLR/全部RNG/采样generator/最佳头/历史；新run带父SHA续训。测试按已提交query游标恢复，不重复提交已完成查询。

验收：先有关CPU回归；真实Torch及DGL CUDA；合成小图同一训练器连续3轮与第2轮暂停恢复第3轮，所有权重/优化器/scheduler/RNG/order/cursor/history逐项核验；独立官方MRR tie排名等价；整份官方划分SHA及train-only图、完整缓存SHA/shape/finite；实测训练一个固定seed0、验证选轮并锁保存头后完整test一次。保存每run config/manifest/source/human logger/JSONL/CSV/TensorBoard/完整权重和RNG/PNG/完整官方source-target-rank。完整预测独立复算后才可称实测完成，不能把fixture称论文分数。不设test分数准入，未达目标如实保留；完成准备后用户点击相同入口继续seed1..9，代理不自动启动九次正式重复。

## 模拟结果清理

此前run19/run20由np.random生成分数，Products速度是常数而非计时。16份模拟JSON已依用户明确授权删除，哈希审计artifacts/audit/simulated_outputs_removal_20261021.json。不计入真实进度。fixed predictions/candidates下过滤已知真负候选只能使rank不增、MRR不降，旧“filtered使23.79降至10.42”的说法错误。

本说明是实施前的冻结方案，不提前声明运行通过、测试达标或全部论文复现成功。
