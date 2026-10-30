# 当前快速质量试验入口

在PyCharm运行 `scripts/run_fast_graph_screen.py`。当前已启动，在恢复中的同一个进程结束前不要再开第二个训练。停止时用 `scripts/request_fast_graph_pause.py`，等待SAFE_PAUSE后再运行同一入口；已完成且哈希核验一致的作业跳过，未完成图编码/分类头从一致状态继续。

复用原八轮固定子集联合模型；Ele-fashion不重训，无额外预训练、无累计24小时截止。先比较修正图的验证集，再锁定处理方式、完成一次完整测试。测试分数不选配置。保留所有官方训练/验证/测试节点，不通过缩小测试集提高分数。

当前实测：Wiki-CS seed=0/split=0测试77.6125%，论文78.57%；Arxiv seed=0测试70.9154%，论文71.66%。这与此前低分的CLIP验证对照不同。单次结果接近不是完整论文统计复现成功。Goodreads-NC尚未确认，不能填入猜测分数。

图处理依据：Wiki-CS的DGL加载器以及MM-Graph的NC GNN入口包含补反向边/去重。精确CAME作者加载脚本未公开，变体和来源在每个run中显式记录。不会把NC图处理套到LP拓扑，后者继续保留训练边及官方反向重叠审查。

Goodreads-NC推理批量128通过完整模型的CUDA压力及8/128向量一致性校准，实际峰值约2.39GiB。仍是FP32、完整邻居和完整官方测试，改变的是执行批量。临时自有嵌入最多3GiB；验证与确认共享一次编码，确认后清理派生缓存。原始资产、模型、旧实验日志与预测不删除。

结果目录从 `artifacts/workflow_state/fast_graph_screen_current.json` 获取，含 `results.json`、`results.csv`、`comparison.png`、逐阶段日志和测试前锁定记录。每个子run含完整配置/环境/代码/数据/父哈希、logger、CSV/JSONL、TensorBoard、RNG检查点及预测/曲线。

审查三项NC后，运行 `scripts/run_remaining_single_seed.py`，对Products、三项LP、两项KG各做seed=0。这些未知成绩必须实测再审查，不承诺预先达标。其后再安排正式重复、few-shot及消融。
