# Products校准RAM暂停修复

日期：2026-10-22，Asia/Shanghai。失败run为20261021T231307Z_products_fas_218254_s0_dbe52c。原入口在校准小图和packed-head fixture之后、4096节点测速之前触发RAM保护；没有完成校准、表征缓存或正式评测。已用99.4081938000163秒活动预算保留，不从零计时。旧运行和修复前源码/release/账本快照保留。

代码中发现整张123718280边图先从compact CSC构造DGL图并转int64，随后adj_tensors可能分配全量edge IDs。采样算法只需要CSC的ptr和neighbor行，根本不需要全量DGL图或edge ID。原失败日志没有记录available/private数值，不能声称已量得失败时峰值，也不能排除其他应用内存压力。修复前实际系统可用内存约2.63GiB。

修复为读取已校验、只读mmap的int32 CSC数组，只有选中邻接位置和小批DGL blocks转换int64；模型、FP32、1024维、fanout10/10、无状态边位置选择、自环语义、官方全部划分、十独立头和两小时准入均不改。小批神经运算必须Torch和DGL CUDA。此为结构内存优化，不是神经计算CPU回退、降维、缩测试或换PPR。原raw/CSC/CLIP/精确前缀均不改、不删除。

内存保护仍要求至少1GiB可用、进程private不超过物理RAM的80%。现在触发保护时记录实际available/private/RSS、阈值及持久化证据，日志解释原因。外层将安全暂停视为正常已保存状态，记录calibration=paused、累计活动秒数保留，正常退出，不再向PyCharm抛TrainingPaused回溯。真实故障仍须失败，不吞异常。

17项CPU检查通过：同一完整CSC上新旧采样inputs、src/dst ID、边序逐位相同，涵盖平行边/自环/孤立节点；packed梯度/Adam恢复、行游标校验、未放宽RAM阈值、诊断落盘和控制器安全暂停正常返回。报告artifacts/audit/products_ram_repair_cpu_20261022.xml。这些不能代替实际CUDA、吞吐、前缀保真和显存验收。

后续先以同一run24入口--prepare-only运行真实校准，锁内串行，不启动十头正式训练。只有校准通过，用户参数留空运行同一文件才继续正式工作量。若吞吐或数值保真未达门槛，如实停止报告；内存修复不证明两小时或论文准确率目标已经实现。
