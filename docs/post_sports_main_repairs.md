# 四项主任务诊断计划（2026-10-15）

目标：优先处理Cloth、WN18RR、Goodreads-LP、Products，绝对差2个百分点是交付目标，不是保证。Sports正在由用户正式训练，不修改其冻结源码/配置/清单，也不启动第二个GPU任务。其全部十seed完成且全局GPU锁释放后，单次串行监督入口` scripts/run_post_sports_main_repairs.py `才执行以下有限诊断；正式重复由用户训练。本入口不是定时自动化。

Cloth：最新完整test MRR×100=26.1155809836，论文28.88，差-2.7644190164pp。固定MMGraph基准代码的预测头有两个Linear/LayerNorm/ReLU隐藏层，而当前Cloth是一个Linear/ReLU隐藏层。这是明确结构差异，尚未证明是性能根因，也不是CAME作者确认。只做一个匹配预测头候选，1024维、冻结encoder、图、全部划分、seed0、K4负采样、lr=.001/batch4096/300上限/10patience全部保留。先用实际Torch+DGL CUDA和seed1完整optimizer/RNG暂停恢复逐位一致验收，再完整validation；至少比原0.294638596907高0.005才锁保存权重test一次，否则拒绝且不做新test。详见configs/cloth_predictor_structure_20261015.json。

WN18RR：现test77.9834077856%，论文81.97，差-3.9865922144pp。继续此前未完成的固定strong-Wolfe线性头候选：完整checkpoint已核验261步/next262、config/data/optimizer/RNG/history和有限数值，保持500步上限/100patience及原参数。新建run引用父SHA30e81df4dfb649953979ca1151674a1ef92afb46806870b986aca2341e3b775d。只恢复WN，不恢复旧LP队列、不清除旧优先暂停；validation比固定参考至少+0.5pp才锁折回线性头并test一次。源/实际CUDA验收既有且重核验，结果可能不改善。

Products：完整2449029节点/1024维FP32/官方2213091测试节点保持。实际全邻居数值及容量校准通过，但不代表全部批次容量/时间已知。完整推理缓存新增10031222784 bytes；分块精确推理、逐段游标与前缀SHA提交，同未提交游标OOM减批重算，batch1仍OOM则明确停止，无CPU模型计算回退。分类头2000上限/100patience，官方validation选最优头，保存锁后完整test一次。单seed结果未知，不能提前交付达标版。

Goodreads-LP：已独立核验MRR×10023.7908054318，论文10.42，高13.3708054318pp；官方150固定负候选、全部1710677测试query及反向非重叠子集已检查。当前没有足够证据确定作者的负例生成/预测头协议，优先审查可比性；不虚构1000负例、不换指标、不缩测试集或人为降分。此项尚未修复，不能因数值更高便称复现成功。

资源：串行单GPU，Cloth/WN不新增编码缓存；诊断头/验收额外预留512MiB，Products额外约9.34GiB。CPU预检已计入全部新增需求并通过项目40GiB/外缓存30GiB/磁盘余量3GiB守卫；启动时重检查。作者原文件、官方raw、已完成五项正式重复与Sports全部结果均保留。

验收：模型选择仅使用train/validation，固定seed0，不依据test差距搜索配置/seed；独立重算全部官方预测/排名、日志CSV/JSONL/TensorBoard与保存头/父SHA，图需人工目视后才交付。当前六项针对性CPU检查通过，Cloth新预测头的实际CUDA验收尚未进行。四项未知细节/工程适配和未达标结果按原值报告，禁止宣称整篇成功。
