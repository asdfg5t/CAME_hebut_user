# 第07项：Sports正式下游重复训练

PyCharm点击 `scripts/run_07_sports_repeats.py`。用户默认Python39已验证能自动转交项目`.venv-came39`；不需要手动配置参数。当前Cloth诊断占GPU时会提示`GPU_BUSY`并退出，未启动重复作业；Cloth结束释放GPU后再次点击。

第07号是本项目的主任务交付顺序，接前四NC和第05项FB之后，为WN保留第06号。不是旧1471清单的位置。

已完成且独立核验的seed0：官方完整37413测试查询，每查询150个固定负候选，MRR×100=35.6955245660；论文37.68，差-1.9844754340个百分点。只复用这个seed0，新增固定seed1-9各一次训练，记录全部结果的均值与样本标准差，不删低分、不选seed。十个下游seed共享同一预训练编码器，不能说成十次独立预训练。seed0接近2pp边界，不能预先保证十seed均值也在2pp以内。

固定训练协议：同一冻结1024维CAME编码与train-only双向去重图；Hadamard输入、两隐藏层1024宽LayerNorm预测头；Adam学习率0.0005、batch512、最多60完整正边轮、验证10轮不提升提前停止；每训练正边采样4个仅按train及其反向已知正边排除的非边，使用当前头无梯度打分取最高一个；联合平均BCE。验证与测试仍使用官方完整150固定负候选。全部按validation选最优保存权重、冻结SHA之后test一次；测试不选参数或seed。作者没有发布LP完整训练细节，本负例/头协议是有记录的工程重建，不能宣称作者操作完全相同。

独立交付清单：`artifacts/remaining_main_pilots/two_lp_priority/verified_job07_sports_delivery.json`。Cloth将使用独立清单，不修改本清单，防止后来交付影响本项续跑。

账本：`artifacts/workflow_state/verified_job07_sports_delivery_repeats_current.json`。每个唯一run保存人类logger、JSONL、CSV、TensorBoard、训练PNG、官方验证/测试排名和独立核验。大权重位置由`configs/storage.json`和每run的`checkpoints/location.json`确定。新增9个seed预计头权重与记录在1GiB预留以内，40GiB项目/30GiB外部缓存/3GiB空闲守卫保持。

再次点击会核验并跳过完整完成的作业，从最近完整轮的optimizer/RNG/采样状态恢复未完成训练，写入新run和父检查点SHA；未保存的轮重算。PyCharm强制停止可能丢失当前未保存轮，不能保证任意batch无损续跑。若已有测试排名但确认阶段不完整，入口拒绝盲目重测，需先核验保存结果。

交付验收：完整官方排名/ID/MRR/Hits/反向子集已独立重算；CSV/JSONL/TensorBoard与最优第53轮一致；保存头/父SHA和PNG已核验。相同难负例实现已通过实际Torch+DGL CUDA、正确候选选择及非零seed完整断点恢复测试。基础Python39 `--dry-run`通过且新GPU作业0；两个训练策略防混用检查通过。正式重复训练由用户执行，代理没有启动。
