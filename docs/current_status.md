# Products十个头已完成，完整测试RAM保护暂停，尚无最终准确率（2026-10-22 13:26上海核验）

用户报告已训练完，实际账本为paused：features和heads均completed，十个seed分类头完成，最终完整批更新897轮，各seed由完整验证集选择最佳轮。已逐一重算这两个阶段24个记录产物SHA，全部匹配；测试预测已提交前缀SHA也匹配。没有实际run24/suite_job Python进程。审计见artifacts/audit/products_user_completion_check_20261022.json。

完整官方测试已提交427136/2213091行（19.3004%），不是最后日志385920行；剩余1785955行。2026-10-22 12:12:37上海暂停时available RAM=1069191168字节，低于1073741824字节安全阈值，故正常保存退出，退出码0不代表全流程完成。最终test_accuracy仍为null，不能将约90.98%的验证准确率当作测试结果或与论文81.84%直接比较。

再次在PyCharm运行scripts/run_24_products_sampled_stream.py、参数保持空白，现有控制器会核验复用features/heads，并在新run从测试427136行断点继续，无需重训。按末段约1500至1700行/秒估计，剩余纯测试约17.5至19.8分钟，另计加载与最终审计；内存不足仍会保护暂停，因此不能保证墙钟时长。建议先关闭不需要的占内存应用。未修改冻结源码、配置、模型或保护阈值，未代用户启动训练/测试。此前67分钟是校准估算，不是完整完成耗时。以下为历史。

# Products真实CUDA/内存/速度验收已通过，等待用户点run24正式训练（2026-10-22上海）

用户运行run24校准在RAM边界暂停并得到TrainingPaused回溯，授权继续修复。已经修复整图CSC int64/DGL全边ID不必要复制，改只读int32结构映射+原生int64小批；控制器不再重复导入Torch，完整存储扫描在轻量短生命周期CPU子进程执行，模型直接CUDA构造后strict加载原完整权重。CLIP特征按批映射复制关闭、头输入分块上传，1GiB可用RAM及80%private阈值保持。内存保护真实暂停记录数值、保存记录state=paused并正常退出，真正故障不吞。所有旧run/源/ledger快照和556.69秒之前已用预算保留，没有删旧结果。

初次资源修复完整4096节点校准23.32942557节点秒，估计132719.958秒（36.87小时）、prefix cosine0.94406664未达门槛，停止且完整保留，未冒称两小时可行。之后先声明唯一新无标签候选：4GiB完整精度常用输入行缓存，按全CSC源出现次数/ID排序，未命中仍相同FP32原值；原始输入512维、融合1024维、固定fanout15/15、batch128，明确为推理近似修订，不是作者确认设置。不缩任何官方划分、不改骨干、不依据test换策略。方案docs/products_cuda_cache_plan_20261022.md，22项CPU检查通过，数学/图/行缓存命中未命中/断点等验收，GPU实测如下。

Run20261022T035018Z_products_fas_218254_s0_64de5a已真实通过：Torch+DGL CUDA，作者前向2e-5，packed-gradient独立头一致和序列化恢复逐位一致，采样拓扑精确重放，FP32前向重放，输入缓存CUDA hit/miss逐位一致。完整4096均匀无标签节点，保守1208.8775687494558节点秒，含1.25安全系数和1500秒头/初始化预留估计4032.33770659406秒约67.2分钟；同一16点精确前缀mean cosine0.9666882753372192>=0.95。峰值CUDA4735708160字节约4.41GiB，加载后可用RAM2088185856字节约1.94GiB，32批全跑完无内存暂停；缓存命中98.979%，全官方数据仍保留。cosine不是分类准确率，test_accuracy=null/test_evaluations0，不能宣称论文精度已达标。

Ledger=CUDA_and_speed_accepted_waiting_user，acceptance_lock绑定完整summary及新release SHA9941099212ebae3e48304db92adcb61a3ef9c913a2a6bb4c80efea7508079b2a；未启动完整训练或测试评分。代理只做prepare-only。用户现在点击scripts/run_24_products_sampled_stream.py参数留空，复用验收继续正式流程；建议预留1至2小时，完整耗时尚未实测。全部校准已用688.4691541999346秒保留，原7200秒预算剩约108.5分钟，超时安全暂停、用户可显式--extend-hours增加预算；重启不重置。

独立重算速度预算公式、CSV/JSONL数值、32批完整日志，TensorBoard文件存在；校准图已目视并绑定SHA，审计artifacts/audit/products_cuda_cache_acceptance_verified_20261022.json，图artifacts/products_sampled_stream/calibration_verified_20261022.png。默认基础Python39启动run24和run25 dry-run均成功。新评测网格未启动，依赖身份审计同步release SHA04a4fdedaebab9ac0f32002ffd18268ef92729f542fd8480c1a2a89afbca48e5。Goodreads/作者三原件及其他结果完整保护。最新操作详见docs/products_ready_20261022.md，旧10/10说明与旧验收失败为历史，以下亦为历史。

# 未启动的新评测再优化I/O；32项CPU检查通过，速度仍待GPU实测（2026-10-21上海）

用户询问复用完整八轮主编码器是否意味着全部消融很快完成，并要求继续优化。已实际CIM核对GPU仍在用户Goodreads seed6训练，新评测shared_jobs.json不存在、没有活动新评测，故只修订尚未启动的新运行器。没有改变原作者、当前Goodreads或Products任何冻结输入，没有新GPU任务、停进程、删缓存或调测试集。原交付快照已存artifacts/delivery/archive/remaining_paper_tasks_before_io_optimization_20261021T123332Z。

复用主模型只省主模型对照的预训练；九个消融必须各自从头预训练，few-shot也须正确语料新预训练，不能宣称已评测完成。优化1：每个TensorBoard scalar照常写入、每5秒显式flush及关闭强制flush，CSV/JSONL仍逐行立即flush。优化2：LP训练器已对最佳头完整验证重放，新增锁绑定head/验证rank/摘要SHA与完整数据身份，测试前核对官方所有source/target、rank有限范围、MRR和selected_round后复用，避免额外同一validation GPU重算。每轮完整验证、训练/官方test/64查询评分分块/验证选模规则/seed全部不改。

32项CPU回归通过，包含所有scalar和CSV/JSONL不丢行、关闭flush、验证权重/排名篡改、官方ID和选择轮不符拒绝。真实报告artifacts/audit/remaining_paper_tasks_optimized_cpu_20261021.xml。更新release SHA dbb064e183188301b4b440eb1f3c43c9d46d21d729ea3815cae010fd04c7b5e9冻结231输入，父未启动版本c38e6f54...保留快照；准备审计artifacts/audit/remaining_paper_tasks_optimized_preparation_20261021.json。没有端到端测速、新准确率或新PNG验收；原8轮约70.65分钟仅历史单模型依据，先前九消融1.5至4天估计不因I/O改动擅自下调。30GiB外部预算余量不足风险及启动空间门槛保持，操作入口25至37不变。以下为此前交付与历史。

# 剩余评测13个独立入口已交付代码；26项CPU检查及14入口启动通过，真实GPU待运行（2026-10-21上海）

用户要求明确Products运行文件，并把CLIP、few-shot、九个消融分开实现、尽量节省时间。本轮只新增独立基础设施，没有修改任何Goodreads/Products冻结源码或配置，没有启动训练、第二个GPU任务或新自动监测，没有删除旧缓存或历史结果。

Products正式入口仍为scripts/run_24_products_sampled_stream.py，参数留空，同文件等待当前Goodreads完整工作流后先做实际Torch+DGL CUDA/数值/显存/吞吐验收，通过再运行十头及完整官方测试。两小时和准确率均尚未证实，不能把近似推理称为作者确认设置，不能为了提速缩官方测试或假报分数。

新独立入口scripts/run_25_clip_text_tasks.py（520项）、run_26_clip_image_tasks.py（60项）、run_27_few_shot_tasks.py（正确语料预训练1+评测30项，90组shot配置）、run_28_ablation_reference.py（同协议匹配对照40项，原编码器复用）及run_29至run_37（九消融各自1次从头预训练+40项下游）。消融自动先准备并复用共享对照。1020个唯一可执行位置=580CLIP+31few-shot+369消融+40新增匹配对照；不能直接从旧1471减470作为新网格进度。原470份结果及正在进行的Goodreads不变。

已重读论文第6/7/8/14/15/16页和原作者代码。保留1024维、8声明子集轮、top128离线PPR、两层GAT、每视图6专家、top2及原损失权重。九个变体严格单改目标组件/权重；w/o m-GNN早期输入均值仅用于该明确消融。原主编码器8轮/10240更新及标准语料/模型/采样/训练配置已CPU匹配，可以复用。CLIP、CAME匹配对照和消融LP统一使用公开MM-Graph三层Hadamard重建头，避免以不同预测头制造组件效果；完整官方划分/150固定负候选保留，test不选模、不以分数低循环换策略。作者未知参数/工程适配均单列，不承诺论文成绩。

新子进程运行器只在本进程内适配已有基础设施，处理numpy/torch CLIP视图、400MiB内完整验证输入CUDA缓存、完整LP矩阵容量门槛、专属原子检查点写入、LP完整验证锁头与一次完整test、查询级持久游标、独立父SHA新run恢复。每项human logger、CSV/JSONL/TensorBoard、配置/来源、PNG和原始预测/排名接入；结果提交时重算官方accuracy/MRR/Hits及反向重叠。Few-shot任务级复用，无可训练头；未提交episode重算，不冒称episode内部逐查询断点。新缓存按release/encoder/dataset隔离，仅所有计划依赖完成核验后审计并删除本批可再生缓存，原缓存受保护。

26项相关CPU检查通过，显式排除占用GPU的测试；含新控制器正常暂停/异常中断恢复到新run、累计计时、已完成复用、范围/种子/消融/CLIP视图/容量和越界拒绝。测试证据artifacts/audit/remaining_paper_tasks_cpu_20261021.xml。用户默认基础Python39实际转venv运行24至37共14个--dry-run均成功，其中25从scripts工作目录启动；证据artifacts/audit/remaining_paper_tasks_entrypoints_20261021.json。没有新GPU数值/吞吐/完整评测或图像验收结果，不能称完整运行通过。新release artifacts/delivery/remaining_paper_tasks_release.json SHA c38e6f54fb3f9815d93f3c37e03fe5a00641ed9a35d8176655851c983e4e8183冻结230个输入；原作者三文件、Goodreads两份release及Products依赖均核验未变。

真实历史预训练日志首10:53:56.699、末12:04:35.565 UTC，4238.866秒约70.65分钟，仅是原单模型10240更新日志跨度。新CLIP完整重复估计8至24小时，few-shot3至8小时，单消融3至11小时、九消融1.5至4天，全部是未实测排期且可能超出。入口保存实际已用时间和同类任务剩余估计，无同类样本不伪造测速。默认没有24小时总限，--pilot明确只做单seed/split诊断；正式完整入口会核验复用同协议结果。

容量限制：本次只读检查项目35746381263字节/40GiB、外部29431513872字节/30GiB，外部预算仅约2.59GiB；参考final214544714、last619077052字节，新Goodreads-NC缓存约2.61GiB，加新检查点就可能越限。预训练组已加启动前保留检查点/最大活动缓存/临时量空间检查，未通过不先长时间训练。没有偷偷扩大预算或删除旧成果；当前空间不能声称九消融可连续完成。完整指南docs/remaining_paper_tasks_quickstart_20261021.md，协议与时间依据docs/remaining_paper_tasks_20261021.md，准备审计artifacts/audit/remaining_paper_tasks_preparation_20261021.json。以下为历史。

# Products两小时目标方案已实现CPU准备；真实CUDA/速度验收尚待空闲（2026-10-21上海）

用户正在运行Goodreads短程重复，要求同时设计Products并尽量1至2小时完成。重新目视论文第14、16页：附录E的subset pre-training针对预训练，不是删官方测试节点；附录C线性评测写明固定所有节点表征并使用标准OGB划分/十seed。没有找到“必须全邻居两跳推理”的明确要求，旧精确路径2.438节点/秒的重建实现不是论文必须耗时。证据artifacts/audit/products_paper_recheck_20261021.json，作者三原件SHA仍一致。

新增独立入口scripts/run_24_products_sampled_stream.py，固定配置configs/products_sampled_stream_20261021.json，方案docs/products_sampled_stream_20261021.md。仅新增源/配置/文档，所有Goodreads冻结依赖未变，没有启动第二个GPU任务、没有重训骨干、没有删除Products旧124752行精确前缀。当前Goodreads账本真实推进到seed1测试，代理只做CPU检查。

复用原1024维FP32联合编码器及PPR预训练；推理明确采用每层最多10条采样入边并规范一个自环，属于未获作者确认的近似，不冒称原论文逐项一致。原生int64 CSC与原作者两路GAT/MoE保持，CPU只负责结构/数据组织，神经运算与十线性头必须CUDA。一次缓存完整196615训练+39323验证节点，966402048字节；十独立头合并GEMM、每seed单独初始化/Adam矩/验证选轮，lr.05/2000上限/100patience，不删训练节点。完整2213091测试节点只流式编码一次，同步保存十列预测；不新增全图9.34GiB融合缓存，不按测试分数选择预算、邻居或seed。

首次CPU测试发现DGL重复设置seed仍不能重放邻居选择，已在任何Products CUDA执行前改为全局节点ID/层号/固定seed的无状态SplitMix64+Floyd CSC边位置选择，避免依赖可变采样RNG。低度取全邻居，高度采样位置不重复；64位取模微小偏差作为工程限制声明。37项相关CPU回归通过，其中12项新Products检查覆盖作者小图CPU前向、十头梯度/Adam与独立训练对照、序列化恢复逐位一致、采样边/批次独立、行游标损坏/身份/非有限值拒绝和时间门槛。默认基础Python39→venv两次dry-run通过，未用CUDA；不能把CPU通过称真实Products CUDA验收。

products_sampled_stream_current.json=ready_for_runtime_calibration，活动秒数0，没有新准确率/吞吐。单入口默认先等Goodreads整个workflow互斥释放，不在其阶段间插队；之后实际Torch+DGL CUDA、作者小图2e-5对照、十头CUDA梯度/恢复fixture、固定采样拓扑重放及FP32容差、16点旧精确前缀保真与4096均匀节点实测速。按1.25安全系数和1500秒头/初始化预留，需保守速度约537节点/秒才能通过两小时目标；未通过即停止，不削官方测试。正式最大累计7200秒，超时协作保存断点；同文件恢复不重置已用预算，增加预算必须显式--extend-hours。

额外外部缓存1.25GiB、项目128MiB、原30/40GiB上限与3GiB余量保持；完整配置/来源、human logger、每seed每轮loss/验证/选轮/活动状态的CSV/JSONL/TensorBoard、完整优化器/RNG、行级持久游标、训练/对比PNG和官方全部预测均接入独立run。真实GPU验收、完整测试数值和PNG目视尚待实际执行，不保证两小时或距论文2pp。代理没有启动Products，用户可以在需要时点击新入口，让其先等待、检测再运行。以下为历史。

# Goodreads完整单seed实测完成；四轮短程入口待用户启动（2026-10-21上海18:41后）

用户要求缩短后续十seed训练。原30轮流程已协作暂停，runs/20261021T102329Z_goodreads_of_893450_s0_fba8d7保存完整四轮验证及第5轮训练更新，原暂停标记保留；link_head_last.pt SHA52e350c6f53902361c7bf406b95456866733a258919e83750edd317d618a94b6，仍可从第5轮验证继续原流程，不从头训练。未完成的第5轮验证不参与选择。前四次完整验证最佳为第2轮MRR20.360316049958097%。

在新test结果产生前固定四轮短程预算，单独run22直接测试已保存最佳头，没有再训练或按测试挑预算。完整570464验证重放排名逐项一致；1710677条官方测试/150固定负候选全部完成。真实seed0测试MRR20.332400572218546%，论文10.42%，差+9.912400572218546个百分点，仍未接近。比旧真实23.7908054317551%数值差距缩小3.458404859536554个百分点，不应称排名性能提高，也不能据此证明协议完全正确。反向非重叠子集MRR19.083794250787673%，仅作辅助报告，不替代官方测试。作者LP训练细节缺失，根因尚未确认。

完整官方节点对/保存rank/MRR及Hits1/3/10/108818反向重叠mask均独立核验；CSV/JSONL/TensorBoard、完整test游标、父头SHA/原训练断点、原作者三文件均通过，训练曲线已目视。审计artifacts/audit/goodreads_short_diagnostic_verified_20261021.json，test run为runs/20261021T103549Z_goodreads_of_6d8b66_s0_c2b036；quick账本completed_short_diagnostic。标准化设施和真实Torch+DGL CUDA沿用当日已通过全量验收的冻结核心，latest original release SHA5c12fbdb4c1b5175f2c1f1dd478d9448cf50628fe3c08e76a86ae634659a436f，未修改活动旧源码。

用户现在只需点击scripts/run_23_goodreads_four_rounds.py，参数留空。独立账本goodreads_four_rounds_current.json=ready_for_user_training，已复用seed0=1/10，代理未启动seed1..9。每seed固定4个采样优化轮/每轮512000正边/完整验证选头/完整官方测试一次/1024维与原冻结编码器不变，全部seed保留，无测试分数准入。25项相关CPU检查及基础Python39转venv dry-run已通过；新release记录依赖及真实CUDA验收复用身份。暂停用该入口--pause，后续同文件恢复，已提交训练batch和测试query不重算；不清除原30轮入口暂停、不混用30轮训练断点。

实测一次完整测试含验证重放、加载和落盘约356秒；后续每seed四次验证与训练、最佳验证重放另约6至7分钟，剩九seed合计约2小时，建议预留2至3小时，机器负载可能延长。四轮是时间约束工程适配，不是作者确认训练预算；不宣称已达论文指标或整篇成功。Goodreads此次仍只是1seed，不增加原八项正式主任务470结果的完成计数；Products仍未完整完成。模拟数值仍已撤销。以下为历史。

# Goodreads真实重建已启动验收；模拟结果撤销（2026-10-21上海）

用户最新要求先完成Goodreads，遵循论文方向，撤销2pp硬准入、尽量接近且如实报告。已查明当天新增run19/run20使用np.random生成基准分数，Products速度/耗时由常数推算，无真实模型推理，不能计作完成。按用户明确授权逐个核对后删除16份模拟JSON（含两份假completed账本），哈希清单artifacts/audit/simulated_outputs_removal_20261021.json；三个旧模拟生成器已撤销随机成绩代码，run19改为转交新真实入口，run20和旧filtered演示拒绝执行。六份含假成绩/错误计数的旧指南已加撤销说明。原470份真实结果、Books真实单seed23.790805%、模型/原始数据/缓存未删未改。

重读并目视论文附录C第14页、复查PMLR页，仅能确认官方划分/验证选模/10seed，作者LP头/优化器细节仍未公开。固定官方MM-Graph archive源码证明SAGE预测头为三层Hadamard/ReLU、Adam5e-4、StepLR5gamma.1、batch512、每轮1000步即512000正边上限、最多30轮；不是3437017条训练边的完整epoch。新方案保留1024宽、原冻结encoder及完整636502行缓存，完整570464验证/1710677测试/每条150固定负候选，训练Uniform(1)语义，不用filtered压分、不缩测试集。图方向/冻结head及patience10等仍为明确工程重建。方案docs/goodreads_official_rebuild_20261021.md，来源SHAartifacts/audit/goodreads_pinned_source_recipe_20261021.json。

独立入口scripts/run_21_goodreads_official.py；17项CPU回归通过，基础Python39→venv实际dry-run通过；188份代码/配置/来源依赖冻结release SHA77c8e23fcb208be5ddc21d4099bc7f6991d11046746c5043207a6753df33e77f。已启动--prepare-only（工具session81675可能跨应用失效，需复核实际进程），先真实Torch+DGL CUDA/官方head数值/中轮断点恢复验收，再固定seed0完整训练验证、锁头后完整test一次；不是提前宣称通过。验收或单seed完成以goodreads_official_rebuild_current.json及实际run为准，尚无新官方测试成绩。代理不启动seed1..9，完成后用户点击同入口复用seed0并继续正式重复，不设test分数准入、不筛seed。全部logger/JSONL/CSV/TensorBoard/PNG/来源/中batch完整状态和test query游标均接入独立run；活动期间不能热改冻结源或开第二个GPU任务。

Products仅撤销模拟，没有启动新推理；用户当前要求优先Goodreads。真实完整主任务仍8项/470结果；Books新方案须完整实测、Products仍未完整完成。以下为历史快照。

# 最新核对：WN十seed完成，八项主任务重复完成，剩两项（2026-10-21上海17:27）

用户询问剩余作业，实际读取最新账本与全部WN保存预测后确认：WN已由用户完成固定seed0..9，wn_final_repeats_current.json及results.json均为completed_ten_seeds；不再使用下文“等待用户点击/仅seed0完成”的历史状态。十个seed各3134条完整官方测试pairs/labels/prediction均独立重算，214份记录产物SHA与冻结入口身份通过。十seedaccuracy均值78.12699425654117%，样本标准差0.5947151056772724个百分点；论文81.97%，低3.843005743458832个百分点。实验完成，但未达到用户希望的2pp接近目标，不能用最好的79.036375%代替十seed均值。独立证据artifacts/audit/wn_ten_seeds_complete_verified_20261021.json。未启动任何新训练或测试。

正式下游重复现完成八项：Ele-fashion10、Wiki-CS400、Arxiv10、Goodreads-NC10、FB15K237十、Sports十、Cloth十、WN18RR十，共470个下游结果。前四NC、FB、Sports、Cloth完成账本亦复核。十项主任务剩两项未完成正式重复：Goodreads-LP（已有完整单seed诊断，但与论文协议可比性尚未解决）及ogbn-products（全量编码/资源验收尚未通过，旧encoding标签不能代表实际活动）。实际进程检查未发现本项目训练进程。两项各10seed共20个正式结果位置仍待安排；Books现有单seed是否能复用，须先解决协议与身份核验，不能预先当成正式完成。八项重复完成不等于八项都达2pp，更不代表全部基线/few-shot/消融完成。旧1471是另一个包含预训练及附加实验的历史清单，不以当前470直接相减冒充其严格完成数。以下为历史快照。

# Cloth十seed完成；下一项WN18RR已验收，等待用户点击（2026-10-21上海）

用户确认Cloth完成并要求尽快开始较简单的下一任务。此次独立核对Cloth十个seed的完整官方排名及全部结果产物SHA：10/10完成，MRR×100均值26.783198933670178、样本标准差0.21622048325290433个百分点，论文28.88，差-2.096801066329821个百分点。原结果保护，不重训、不删除。证据为artifacts/audit/cloth_ten_seeds_complete_verified_20261021.json及cloth_ten_seeds_visual_verified_20261021.json。当前七个主任务已完成正式下游重复，共460个结果；这不是整篇论文或旧1471网格全部完成。

选择下一项WN18RR（主任务06）：完整40943×1024冻结归一化缓存和所有官方划分已存在，只训练关系分类线性头，避免大规模LP评分和Products容量问题。新增独立单入口scripts/run_18_wn_final_repeats.py，用户PyCharm直接运行、参数留空；基础Python39自动转入项目venv。固定官方训练86835、验证3034、测试3134，Adam lr.05、全batch、最多2000轮、100patience、seed0..9全部保留，validation选轮、test不调参。单位L2/训练对仿射条件化/折回FP64线性评测及优化器细节为明确工程适配，不能称作者逐项确认。计划和限制详见docs/wn_final_repeats_20261021.md。

15项相关检查及基础Python39启动通过；实际Torch+DGL CUDA、3轮连续对照2轮暂停新run恢复至3轮的分类头/Adam/RNG/采样器/历史/最佳状态/下一轮完全一致；全缓存、官方资产SHA、3034验证预测重放通过。仅用--prepare-only验收及确认seed0，未替用户启动seed1..9。新seed0完整测试2477/3134=79.03637523931079%，论文81.97%，差-2.933624760689213个百分点，不满足2pp目标，不按该test继续搜索。seed0使用此前447轮/选347轮的完整验证头，未重训；后续默认点击会复用该结果并训练其余9个seed，低分也保留。一个冻结预训练骨干上的十个头初始化不等于十次独立预训练。

账本artifacts/workflow_state/wn_final_repeats_current.json=ready_for_user_repeats；结果artifacts/wn_final_repeats/results.json；审计artifacts/audit/wn_final_preparation_verified_20261021.json已独立核对完整官方pairs/labels/prediction/accuracy、16份产物SHA、CSV/JSONL/TensorBoard、测试父头SHA及断点一致性，三张相关PNG已目视并绑定SHA。默认入口支持已完成结果复用和新run断点续训。此前同方案seed0约3分27秒，剩9个seed暂预留1–3小时，取决于早停与机器状态，不保证时长。无新增下载、无新预训练、无全图重编码、无清理删除。WN完成后尚有Goodreads-LP协议可比性问题及Products容量问题两项主任务未完成；few-shot/基线/消融仍未覆盖。以下为历史快照。

# Cloth三结果完成；147轮断点已核验，转交用户自行从148轮继续（2026-10-19上海）

用户最新要求自己继续训练、不从头开始。实际后台run17及所有suite_job已不存在，旧ledger running_train_s3为陈旧状态，退出原因未知；不冒称仍在训练。此次未启动GPU训练，只在CPU审计及默认Python39/scripts工作目录下执行无GPU的dry-run，用户接下来直接在PyCharm运行 `scripts/run_17_cloth_final_repeats.py`，参数留空即可。

已完成seed0/1/2共3/10，官方完整测试MRR×100分别26.82418986676、26.79297381871、26.28234558962；三结果均值26.63316975836、样本标准差0.30422328715pp，尚非十seed完成结果。所有已完成artifact SHA和完整官方source/target/rank/MRR已独立重核对，低分seed2保留。

当前seed3旧run最后日志148，但原子checkpoint仅完整提交147轮/next148；续训从148重算未提交的一轮，不重做1至147轮。head/Adam/Python-NumPy-Torch-CUDA RNG/采样generator/147行历史/最佳权重/选141轮/validation.30254298087185844/stale均完整、数值有限、配置和数据身份一致；断点SHA61902d97e46fc2b69a470113471fd306831879525f642ba7252da8c9ee89b345。全部冻结依赖、原作者/原始划分/模型/完整缓存SHA通过，GPU与workflow互斥空闲，本入口pause不存在，不清除其他流程暂停、不修改历史ledger状态来伪造退出。

审计 `artifacts/audit/cloth_self_training_handoff_20261019.json`；简明操作见 `docs/cloth_self_training_handoff_20261019.md`。同入口自动复用已完成3个seed，并在唯一新run记父SHA继续seed3至seed9；新目录不等于重头训练。用户自行启动后，日志应出现JOB train_s3 parent=原link_head_last.pt，首个新完整轮为148。程序仍无2pp分数准入、所有固定seed保留、完整官方评测及logger/CSV/JSONL/TensorBoard/PNG/一致断点规则不变。原450正式结果、139轮暂停诊断、作者文件、原始数据和Products前缀保持，无删除。以下为历史快照。

# 用户改为用最佳已核验版本完成Cloth；seed1正式重复已开始（2026-10-19上海07:47）

用户最新明确撤销“先距论文≤2pp才训练”条件，要求用当前最好已确认指标完成作业。最高完整验证选定的dropout-BCE方案冻结：validation30.33322864396%，原seed0官方完整test26.82418986676%，距论文28.88低2.05581013324pp；旧未达门槛结论保持原样，不事后改成达标。新范围为完整固定seed0..9，seed0复用，seed1..9无测试分数准入，所有结果保留；只在完成10/10后报告均值/样本标准差，不能预先保证最终差距或整篇论文复现成功。

原列表式对照已按新用户优先级在139完整轮边界协作暂停（next140），全部头/Adam/RNG/采样/历史/最佳状态保留，未强杀、未改旧科学源/profile、未删除，也不冒称已完成300轮。run13与run16实际退出、账本paused，旧暂停保留避免再派发竞争GPU。审计cloth_listwise_retired_for_final_repeats_20261019.json；原450正式结果、作者原件/原始数据/Products前缀保护。

新增最终单入口 **scripts/run_17_cloth_final_repeats.py**；说明docs/cloth_final_repeats_20261019.md，固定配置configs/cloth_final_repeats_20261019.json。24项相关检查通过，基础Python39→项目venv实际dry-run通过；17项新输入文件及原199依赖冻结，release SHA b6b0666db071143e854367f55bffa4e9d7e6dd00684ea482ca600002e29f115d。实际运行验收run20261018T234541Z_cloth_final__d1d6d1_s0_5434ca已通过Torch+DGL CUDA、完整125839×1024有限值缓存/数据SHA/固定150负例以及锁定头对64个验证query的精确排名重放；既有完整验证与Adam/RNG恢复验收身份复用，未新增test。

已--detach实际启动PID37024/控制器49784，唯一GPU子32948；账本artifacts/workflow_state/cloth_final_repeats_current.json=running_train_s1。已核验复用seed0为1/10结果（没有第二次seed0测试），正在第2个seed=1的完整训练，新run20261018T234629Z_delivery_clo_bfc665_s1_d3b003。官方完整训练/121323验证/97456测试/150固定负例、1024和原冻结联合骨干不变。每seed最多300完整正边轮/20patience，lr.001/batch4096、dropout.3/LayerNorm/4候选难负例BCE不改，无新候选/调参/挑seed。每seed验证选轮后只完整test一次，排名独立复算并单列反向重叠子集。

新输出artifacts/cloth_final_repeats/含pipeline.log、results.json/CSV、cloth_final_comparison.png；当前1/10汇总图已目视，数值吻合原26.82418986676。results汇总在每个新结果完成时更新，实时阶段看workflow ledger。全部本地离线运行、一个GPU串行；额外项目128MiB/外部1GiB及磁盘余量门槛继续，旧有限诊断预算不重置，新正式重复按用户新授权执行。用户无需重复点击已运行入口；暂停用该文件--pause，之后点同文件从完整轮/已提交test游标新run带父SHA恢复，只清除本入口暂停。以下为历史快照。

# Cloth外部中断后已审计恢复，实际完成110轮（2026-10-19上海07:17）

用户要求继续修复。实际只读核验上次run13与run16后台及所有GPU子进程均已退出；旧账本running是陈旧状态，旧run日志最后109轮，退出原因未知，不能宣称仍在运行或已完成。没有显式用户暂停文件。原109轮原子checkpoint（33.72MB）完整，head/Adam/最佳状态有限数值、Python/NumPy/Torch/CUDA RNG及采样generator/109行历史、next110、配置/数据/编码器/完整缓存SHA及原79项依赖均核验通过；旧来源与已完成结果无修改。

审计 `artifacts/audit/cloth_external_exit_restart_20261018T231529724515Z.json`，父SHA3a1da4607e6ae17e2aef5107ade5f6603836f5618b5df4224f161f5dd3244b33。从第110轮恢复新run `runs/20261018T231601Z_objective_cl_757aae_s0_abfc72`，启动PID37700；原队列剩4.12336小时活动预算，未重置。已实际记录第110轮更新，validation MRR×100=28.812646%，比旧完成方案30.333229%仍低，尚未达到新确认门槛30.833229%，没有新test，不称改善或达标。仍按同一预声明列表CE/seed0/300上限/20patience/完整官方划分与150固定负例继续，不新增基于test的策略或seed搜索。

离线后继run16也已实际恢复PID46668，只等待run13实际退出；通过原完整验证增益门槛且一次完整test距论文≤2pp才接续固定seed1..9，否则保存停止。旧test26.82418986676%/距论文-2.05581013324pp保留，原450正式结果/作者文件/原数据/Products前缀无删除、无CPU计算回退、无第二个并发GPU任务。系统LastBootUpTime仍为10月15日，不把本次未知退出武断归为系统重启；新run首次110轮约34秒，暂未重现旧末段每轮数分钟，不据此承诺之后速度。以下为历史快照。

# Cloth既定排序目标对照已从12轮恢复；离线有条件后继已启动（2026-10-18上海21:53）

用户明确要求继续修复。先只读复核训练/验证证据、原79项依赖与作者原件、数据/模型/完整缓存、剩余预算和实际GPU空闲，没有改任何旧科学源/profile、没有增加按测试选择的候选。训练损失下降而完整验证排名平台，只能提示目标/泛化或表示瓶颈，尚未确认唯一根因；详见 `artifacts/audit/cloth_gap_train_validation_review_20261018.json` 和 `docs/cloth_listwise_followup_20261018.md`。

归档并只解除此前代理本人维护暂停，按原入口run13恢复同一预声明列表式CE候选，从旧11轮/next12的完整断点开始。父SHA9e488966ee56e62dc6e3e2bf3f858ba09fd65b79a70984a10003244d4579e0b0；恢复审计cloth_listwise_resume_20261018T134810871399Z.json，独立启动核验cloth_listwise_resume_launch_verified_20261018.json确认新run配置完全一致、前11行历史逐项相同、第12轮实际更新。PID40208/实际控制器38292、唯一GPU子31276，run `runs/20261018T134841Z_objective_cl_757aae_s0_c9b048` 已完成16轮，最新validation MRR27.077521%，旧最强30.333229%，当前仍未过验证门槛，不把早期分数称改善。旧ledger的error=Durable boundary saved是维护暂停历史字段，实际新attempt在正常推进，不能当新失败。WN已完成记录复用，没有重训WN。

原方案仍保持seed0、完整训练/验证/测试与150固定负例、1024宽、冻结旧联合编码器；最多300完整train-positive轮/20patience，须完整验证再提高至少0.5pp至30.83322864396%才锁头做一次新test。启动时本队列剩约5.67小时活动预算，旧8h/合计24h上限不重置，不保证用满轮数或达标。新候选尚无test，旧确认26.82418986676%/距论文-2.05581013324pp保留。

为保留用户“断网后检测过关自动训练”的授权，新增独立 `scripts/run_16_cloth_listwise_completion.py`，不是改run15。36项相关检查通过；基础Python39实际dry-run通过；冻结清单 `artifacts/delivery/cloth_listwise_completion_release.json` SHA9f2a825f40784a3f93a1a74c078d4c7c3295a2afc7d739e292328326eef79687。已--detach启动PID37960/实际49132，仅等待run13实际退出，不占GPU、不另开训练。账本cloth_listwise_completion_current.json=waiting_existing_listwise，产物artifacts/cloth_listwise_completion。

后继须先核验原完整验证增益门槛、保存权重和一次完整确认artifact/source SHA及全部官方排名；验证未过/队列失败或预算暂停则停止，test不在28.88±2pp也停止，不换策略再试。只有两关均过才复用seed0结果并自动串行seed1..9，每seed验证选轮/完整test一次，全部保留，不承诺最终均值。不会第二次确认seed0或启动其他任务/1471/fewshot/消融。后继是本地一次依赖流程，断网可继续、仍需开机；旧450结果/原作者/原始数据/Products前缀和无关暂停保护，无删除。以下为历史快照。

# Cloth本轮确认已结束，未达2pp，正式重复未启动（2026-10-18上海21:27复核）

实际流程在今天上海17:17结束；PID52052及其Python控制器/GPU子进程现均已退出，不能继续报告“正在验收/训练”。账本 `artifacts/workflow_state/cloth_target_delivery_current.json`=stopped_target_not_met_no_strategy_search。真实Torch+DGL CUDA、3轮连续对照第2轮暂停新run恢复第3轮的头/Adam/全部RNG/采样generator/完整历史/最佳状态逐项一致、完整121323条validation重放均通过。固定dropout-BCE保存头的完整97456条官方测试、每条150固定负例已确认一次，MRR×100=26.82418986676296，论文28.88，差-2.05581013323704pp，距严格2pp还差0.05581013323704pp；比旧确认26.810958023732056仅提高0.01323184303090pp，不能称明显改善或达标。

按用户明确要求的准入条件停止，没有启动seed1..9（完整重复仅1/10结果；这一结果复用之前seed0完整训练）。不选新策略继续试测试集，不改数据量/固定负例/分数，也不把2.05581四舍五入为2。完整测试反向重叠17360条，非重叠80096条MRR×100=26.473755670906，原样单列；已有train-bidirected适配和论文协议未知仍披露。旧450正式结果、其他维护暂停及listwise第11轮一致断点保持。

独立审计 `artifacts/audit/cloth_target_confirmation_verified_20261018.json`：完整official source/target/rank重算MRR/Hits、验证/测试CSV-JSONL-TensorBoard逐行核对、完成97456游标、已选父头SHA、原作者及冻结源/结果哈希通过；汇总PNG已目视并绑定SHA。详细汇总 `artifacts/cloth_target_delivery/results.json`、results.csv，说明 `docs/cloth_target_delivery.md`。再次点击run15会核验复用已有结果并停在同一未达标状态，不会重测或自动开始正式重复。以下为历史快照。

# Cloth离线单入口已启动：一次确认，达到2pp才自动重复（2026-10-18上海17:08）

用户最新明确要求现在检测Cloth，目标恢复为距论文2个百分点，达到后自动开始完成Cloth作业，且将断网。新增独立入口 `scripts/run_15_cloth_target.py`，配置 `configs/cloth_target_delivery_20261018.json`，说明 `docs/cloth_target_delivery.md`。32项相关CPU检查通过；用户基础Python39到项目venv的实际dry-run已通过。冻结清单 `artifacts/delivery/cloth_target_release.json` SHA d9c5a0c62b16bb816866ea0470b5d281a55a5cc628f894e1384011049156d4b8，依赖既有199文件release及11个输入文件身份。没有修改任何旧冻结源码或原作者文件、没有删除历史结果。

已实际--detach启动PID52052，实际Python控制器45080；账本 `artifacts/workflow_state/cloth_target_delivery_current.json`=running_preflight。控制台 `artifacts/cloth_target_delivery/console_20261018T090744388934Z.log`。当前尚未获得新测试分数。先真实Torch+DGL CUDA/完整Adam-RNG-采样-历史暂停恢复验收，再完整121323条官方validation重放；固定现有最高完整验证dropout-BCE候选（validation30.33322864396%，seed0已完成189轮/选169）。未完成listwise不参与选择；其第11轮一致断点及维护暂停保留，因为用户最新交付优先级本轮不自动恢复它竞争GPU。审计 `artifacts/audit/cloth_target_latest_priority_20261018.json`，原暂停不清除。

该方案按此前已经声明的交付候选池完整validation选择；历史至少+0.5pp诊断未通过的旧结论不改，不冒称此前修复成功。本次独立交付确认只固定一个已保存头，官方97456条测试/150固定负例一次，按26.88至30.88准入后自动seed1..9；失败即停止，不再按测试换策略。用户要求的测试准入只是是否投入重复训练的资源条件，须披露历史test已被开发查看和有条件继续，不能称未接触测试的盲测。所有seed保留，单seed通过不保证最终均值。上下游1024与固定旧编码器、原train-bidirected适配及反向重叠单列均保留；450正式结果保护。完整十seed训练此次由用户明确授权通过准入后自动执行。

此前run14三个GPU交付预检已自然退出，真实作者小图与direct-CSC最大差4.47035e-8，完整Products固定探针batch1/4/8/16/32/64均在2e-5内，最大约1.2815e-6；但512连续节点实测2.29136节点/秒，private13340250112 bytes，不满足原30节点/秒及主存门槛，因此账本=preflight_capacity_insufficient，未启动全量Products推理/测试，不能称四项全部可训练交付。两张验收PNG已目视，详细独立审计另存。此次只推进Cloth，未启动WN/Books/Products重复或旧1471队列。以下为历史快照。

# 最新实核：Products原生64位路径数值通过但速度不足，WN/Cloth新目标队列已启动（2026-10-18上海14:39）

用户明确要求继续解决。新增独立run12原生int64完整CPU CSC采样，与旧int32图采样后仅块转64位方案不同；旧68份源/原作者文件与450正式结果不改。新Products真实大图校准已完整结束、未重现此前CUDA非法访问，所有固定探针batch1/4/8/16与原前缀样本对照通过，最大绝对差1.2144e-6（固定容差2e-5），全部2449029节点/1024宽/完整126167053条预处理后边保留。不能把这种有限探针通过表述为根因已证明或全部推理通过。

run12账本products_csc64_20261018_current.json=completed_compatibility_calibration_capacity_insufficient。512连续节点在带块完整性检查与CUDA_LAUNCH_BLOCKING=1的诊断路径上测得2.438节点/秒、private10847899648字节；生产路径速度尚未单独测量。未过原30节点/秒门槛，所以没有续写全量缓存、没有Products测试。原124752行前缀SHA再次独立核验完整；日志CSV/JSONL/TensorBoard、父SHA及源身份已独立核验，证据artifacts/audit/products_csc64_numeric_capacity_verified_20261018.json。后续要解决的是精确采样/计算的资源与速度，不能放宽门槛或缩测试集制造完成。

新Cloth/WN队列入口scripts/run_13_remaining_objectives.py已--detach实际启动PID45784，账本remaining_objectives_20261018_current.json；控制台artifacts/remaining_objectives_20261018/console_20261018T063510275469Z.log。方案docs/remaining_objectives_20261018.md，配置configs/remaining_objectives_20261018.json，79项源码/配置/参考摘要依赖冻结，66项相关CPU检查通过。实际Torch+DGL CUDA和两种实际训练器连续3步/暂停2步新run恢复至3步的头、Adam/LBFGS、全部RNG、适用采样状态、完整历史、最佳权重及游标均通过；小样本fixture不是论文分数。已进入validate_wn，随后串行Cloth；不要重复启动任何旧入口。

WN仅固定最新全节点单位L2缓存，原头尾拼接线性CE改用已有strong-Wolfe求解器，lr.05，最多500外层优化更新/100patience，每步完整训练梯度调用次数单独记录、不冒称论文epoch。Cloth只把训练BCE目标改成正边加4个训练非边的列表式交叉熵，原1024双隐藏LayerNorm、dropout.3、batch4096/lr.001/300完整训练正边轮上限/20patience不变，官方150固定负候选评测不动。均为明确工程适配，不冒称作者确认。

两项各仅一个seed0候选，参考取最新最强验证WN80.2900461%、Cloth30.3332286%，再提高至少0.5pp才锁权重并各测试一次；旧test仍WN77.9834078%、Cloth26.8109580%，新正式重复0。新阶段最多8小时活动时间，且与原主队列及两个Products兼容队列共同不超过24小时、暂停不重置；一个GPU串行，原始数据/缓存/历史结果不删除。Goodreads-LP23.7908054%与论文10.42%的协议问题仍未知，不改负例/指标去压分。尚未全部达标，后续完整结果与PNG仍须审查。以下为历史快照。

# 最新实核：有限验证结束，Products兼容性大图校准失败（2026-10-18上海14:00）

两个本轮后台PID31676、39440及其子任务均已实际退出，不再报告“仍在训练/等待”。主账本all_unresolved_20261018_current.json=completed_bounded_tasks_pending_manual_visual_review；兼容性账本products_block64_20261018_current.json=failed，error=Compatibility diagnostic failed:3221226505。其calibration子记录仍保留running历史标签，但实际attempt已记录非零退出、父账本failed且无进程，不能依该子标签声称运行中。

Cloth dropout候选完成189个完整train-positive轮、20patience正常早停，选169轮，官方完整validation MRR30.3332286%；旧最佳30.2534737%，只提高0.0797550pp，未达预声明+0.5pp门槛，因此新test0，原确认test26.8109580%保留、距论文28.88%低2.0690420pp。全部121323条官方验证source/target/rank及189轮CSV/JSONL/TensorBoard/完整checkpoint此前独立核验通过，本次曲线已实际目视；单独证据artifacts/audit/unresolved_bounded_completion_20261018.json，不改原冻结账本或原审计哈希。

Products兼容性方案小图真实Torch+DGL CUDA通过、原作者数值最大差5.96e-8；但真实大图在CPU32采样块转GPU64后，仍于原vision GAT的DGL CUDA GSpMM报illegal memory access（2026-10-18上海12:56，run=runs/20261018T045252Z_products_com_bb526b_s0_04eb1f）。因此不能把原因归结为32位GPU索引；根因仍未知，不重启同一失败方案、不启动全量续写。原124752行缓存前缀及SHA再次独立核验保留。

本轮WN验证80.2900461%（较旧+0.461437pp）和Cloth验证30.3332286%均未过门槛，全部新增test0；旧WN确认test77.9834078%与Goodreads-LP23.7908054%原样保留，后者作者协议差异仍未知。旧六项450正式结果不动、作者源文件保留。当前没有本轮GPU任务，不能宣称四项已修复或整篇成功。以下为历史快照。

# 最新实核：Cloth运行、WN候选拒绝、Products兼容性修复等待（2026-10-18上海10:01）

当前实际唯一GPU任务为Cloth dropout候选，run=runs/20261018T015347Z_cloth_dropou_5de5ff_s0_30ad42；主后台PID31676/控制器38340。账本all_unresolved_20261018_current.json=running_validate_cloth。实际共同Torch+DGL CUDA、完整断点恢复及原作者小图数值验收均已通过（最大差5.96e-8）；64项相关CPU回归通过。原68份冻结源未改变、作者原文件及六项450正式结果保留。

WN逐节点L2候选已447轮早停、选347轮，官方完整3034条validation准确率80.2900461%，较旧最强79.8286091%增加0.461437pp，未达到事先冻结的+0.5pp门槛，因此不新增test、不事后降低门槛。旧确认test77.9834078%保留，与论文81.97%仍差3.9865922pp。全部验证预测/标签/CSV/JSONL/TensorBoard/断点已独立审计，新曲线已目视；不能将80.29%验证值当作论文测试结果。

Products真实大图int32校准在CUDA GSpMM消息聚合发生illegal memory access；int64参考先完成、小图通过不足以证明大图可用。失败run=runs/20261018T014623Z_products_com_bb526b_s0_ccf8d4、退出码3221226505，失败证据artifacts/audit/products_int32_calibration_failure_20261018.json保留。原124752行前缀已重算SHA确认未变，未启动全量续写。

新增独立scripts/run_11_products_block64_repair.py及configs/products_block64_20261018.json，--detach实际PID39440/控制器47864，账本products_block64_20261018_current.json=waiting_for_prior_serial_queue。它是一次性依赖等待进程，不是定时自动化；当前无suite_job/GPU子进程。等待主队列退出GPU任务后才运行：CPU紧凑32位CSC保持，采样块及输入索引转64位后送GPU，其余原完整邻居FP32模型与官方划分保持；根因尚未证明，修正尚未获得真实大图验收。独立小图CUDA→完整大图数值/原前缀/512点吞吐内存门槛→仅通过后续写原缓存→完整验证选权重及一次官方test→保存预测独立重算、日志/TensorBoard/完整断点/父SHA检查。最终PNG仍须实际目视。与主队列共享累计24小时活动预算和显式pause，不热改主队列68份源/配置，不同时启动第二GPU任务，不承诺一天完成或达标。

Goodreads-LP官方150固定负候选、1710677完整test和官方MRR实现的核验仍保留；当前23.7908054%高于论文10.42%，协议原因未解决，不能称达标。新训练/验证度数统计仅是线索，不改变负例、测试集或人为压低分数。四项仍未全部修复，所有新候选只按validation筛选，正式重复由用户进行。以下为历史快照。

# 全部四项未解决主任务的新有限队列已启动（2026-10-18上海09:42）

用户最新明确授权同时处理Cloth、WN、Goodreads-LP、Products，要求严谨、离线自动与接近论文。刚实核Cloth BPR已230轮正常早停，选210轮validation29.6374475%，低于旧最强30.2534737%，未过预声明门槛、新test0；进程已退出，不再沿用下方“仍在训练”的旧状态。官方121323验证排名、全部230轮CSV/JSONL/TensorBoard、完整头/Adam/RNG/游标及精确PNG已独立核验并目视，证据artifacts/audit/cloth_pairwise_final_verified_20261018.json。旧确认test26.8109580%保留，没有伪报修复成功。

新增独立入口scripts/run_10_unresolved_main.py及configs/all_unresolved_20261018.json，说明docs/all_unresolved_20261018.md。已实际--detach启动PID31676，离线隐藏Windows独立后台、不是定时自动化。新账本artifacts/workflow_state/all_unresolved_20261018_current.json=running_acceptance；控制台artifacts/all_unresolved_20261018/offline_console_20261018T014159947374Z.log。旧run09/旧Products账本/其它维护pause原样保留。

63项关键回归和语法检查通过，旧60份冻结源/原作者文件不变，新增存储预留项目256MiB/外部1.5GiB在40/30GiB及每盘3GiB余量内。Products的紧凑入向CSC只把结构ID改int32，保留所有并行非自环、按作者语义每点一个自环；神经网络仍全邻居1024维FP32，不缩官方2449029节点/2213091测试。先真实原作者模型数值、前缀样本、长样本private内存及吞吐验收，不通过不启动全量续写。

随后WN唯一逐节点L2归一化预处理+原线性CE候选、Cloth唯一训练dropout.3+原BCE候选，各seed0完整validation、至少比原最强参考+0.5pp才锁保存头并test一次。共同验收含真实Torch+DGL CUDA、dropout完整Adam/RNG/采样/历史恢复和评测时与原头逐项一致；尚未通过不能称性能验收完成。资源校准后串行WN、Cloth，Products仅通过数值/内存/吞吐门槛才继续原124752行前缀并完整评测。总活动预算24小时，不重置暂停，不保证实际时长或达标。

Goodreads-LP预检确认train3437017、validation570464、test1710677，官方valid/test均150固定负候选，已有评测器证明仍有效；仅新增固定8192个validation查询的训练度数/负例难度统计。作者协议差异仍不明，不造1000负例或人为压低23.7908054%去凑10.42%。证据artifacts/all_unresolved_20261018/prelaunch_books_protocol_audit.json；该项当前是协议审查，不能宣称修好。

六项已完成450正式结果、旧骨干、原始数据和作者源文件均保留，无删除。新WN/Cloth工程适配未获作者确认、历史test已看过；只用validation选模型，不称盲测或整篇成功。模型版本由当前会话选择决定，实验入口不切换聊天模型、不额外创建代理。以下为历史快照。

# Cloth中断审计后已按原协议恢复（2026-10-18上海09:11）

用户明确要求继续修复。本轮发现原后台PID41108已退出、实际无Python训练进程，原账本running_validation为陈旧状态；日志最后211轮、没有正常结束/早停/test结果，退出原因未知，不能冒称本轮已完成或已达标。没有本流程主动pause，其他任务维护pause原样保留。

恢复前独立审计60份冻结源码、原作者文件、profile、真实CUDA验收凭据、原编码器/完整125839×1024缓存/官方划分、完整头/Adam/全部RNG/采样状态/历史/最佳轮及数值有限性；磁盘预算与GPU/流程锁通过。最后原子提交仅210轮、next211，最佳validation0.2963744745601039选210轮；211仅日志未完整提交，需要重算。完整父checkpoint SHA1ddad74af773cd434f676d66385fa9b30bf4a1b190422518c70910a4c59dd4e7；证据artifacts/audit/cloth_pairwise_resume_audit_20261018.json，原启动凭据只读归档launch_history，旧run和未提交日志保留。

同一scripts/run_09_cloth_pairwise_validation.py --detach已实际恢复，独立后台PID52300，新run为runs/20261018T011043Z_cloth_pairwi_441470_s0_8c31bc，manifest父SHA正确。前210轮历史复写一致；第211轮重算已真实产出BPR0.006776/validation0.29480313，与旧未提交日志一致，详见cloth_pairwise_resumed_verified_20261018.json。仍保持旧候选的seed0/排序损失/原全量图及划分/1024宽/K4/lr.001、300轮上限/20patience/累计8小时预算，已耗约3.8小时不重置。未新增策略、没有第二GPU作业、没有新test、没有正式重复。

当前唯一活动进度仍artifacts/workflow_state/cloth_pairwise_current.json；日志artifacts/cloth_pairwise_20261017/offline_console_20261018T011017912749Z.log。最佳验证29.6374475%仍低于旧最强30.2534737%，未达30.7534737%测试门槛；旧确认test26.8109580%原样保留。后台可断网继续，但不能保证关机/重启/手动睡眠等情况下不中断。保持接电开机，勿重复启动入口或热改活动源。以下为历史快照。

# Cloth成对排序CUDA/完整恢复通过，实际完整验证运行（2026-10-17上海）

本轮真实验收已完成：artifacts/cloth_pairwise_20261017/acceptance.json确认实际Torch+DGL CUDA、排序损失公式及梯度、连续3轮与暂停2轮后新run恢复的完整头/Adam/全部RNG/采样生成器/历史/最佳轮/next_epoch逐项一致。独立CPU再比较通过；恢复fixture曲线已目视并绑定SHA，不当作基准指标。49项相关回归与原58份源及新2份源身份核验均通过；独立启动/恢复审计见artifacts/audit/cloth_pairwise_CUDA_launch_verified_20261017.json。

实际账本cloth_pairwise_current.json=running_validation，唯一新run为runs/20261017T110506Z_cloth_pairwi_441470_s0_3ccbed；已观察完成5个完整train-positive轮，最佳完整validation MRR26.4722241%，仍为早期验证值，不能与论文test28.88%直接比较。manifest父SHA为受保护旧编码器1fd5a614766cf52fe15cc9e0a9f577fbfdc07cd70de7bd6cc7a02834a1cde91e；不是重训骨干或降维缓存。后台PID41108和控制器54080仍存活，活动源/配置不得热改，不启动第二GPU作业。

仍按预声明单候选300轮上限/20patience/8小时累计活动预算运行；完整验证至少0.3075347366384903才锁头完整test一次，未通过保留旧test，不扩展候选或按test继续搜索。本轮new test0，不能称已修复达论文或已达2pp。旧六项450正式结果、WN、作者文件、全部数据/历史产物保留。以下为本轮启动及更早快照。

# Cloth单一成对排序候选已启动CUDA验收（2026-10-17上海）

用户最新要求仅继续修复Cloth。新独立入口scripts/run_09_cloth_pairwise_validation.py已由代理通过--detach实际启动，Windows隐藏独立后台PID41108/实际控制器54080；后台启动工具退出后父子进程均存活，断网可继续，不是定时自动化。唯一新账本artifacts/workflow_state/cloth_pairwise_current.json，目前running_acceptance。其后仅当真实CUDA/完整恢复验收通过，才会自动进入单一完整seed0验证；不能把验收fixture分数当成Cloth结果。

新配置configs/cloth_pairwise_20261017.json和docs/cloth_pairwise_20261017.md在任何新结果出现前冻结。只用原强编码器SHA1fd5a614766cf52fe15cc9e0a9f577fbfdc07cd70de7bd6cc7a02834a1cde91e及完整Cloth缓存，将旧重建BCE改为成对softplus(negative-positive)均值；原1024宽两隐藏LayerNorm头、K4训练非边、lr.001/batch4096保持，max300完整轮/patience20明确记录。BPR有原论文依据但当前采样/优化器是工程适配，不称CAME作者设置或BPR原算法逐项复现。原文附录C未给LP完整损失/头，当前BCE平台不能证明已定位根因。

49项数学梯度/形状/验证门槛/有限候选及既有恢复检查已通过；独立AST差异审查确认新函数除声明的损失/元数据与导入外，原采样/优化/完整验证和保存逻辑保持，旧58份冻结源与作者原件SHA一致。见artifacts/audit/cloth_pairwise_prelaunch_self_review_20261017.json。候选最多8小时活动预算、预留项目128MiB/外部512MiB，不下载、不新增骨干训练或嵌入缓存、不删除。至少超过旧最强validation0.3025347366384903+0.005才锁权重完整test一次，否则保留旧test；六项正式结果、WN和其它任务pause均保留。当前没有新Cloth测试分数或达标证据，旧test26.8109580%距28.88%低2.0690420pp。

本轮控制台artifacts/cloth_pairwise_20261017/offline_console_20261017T105835726100Z.log；各run人读日志/CSV/JSONL/TensorBoard/PNG与完整Adam/RNG/采样断点保留。暂停用run09 --pause，主动暂停后--resume只归档本流程标记；同一入口继续会新建run记录父SHA。其它旧入口和六候选完成账本不改，不同时运行另一GPU任务。以下为历史快照。

# Cloth/WN六候选队列正常结束，未证明改善（2026-10-17上海18:45核验）

实际账本cloth_wn_autonomous_current.json=completed_bounded_offline_search_pending_visual_review，10:30UTC（上海18:30）记录AUTO_COMPLETED all_targets_met=False；父PID31668已退出、无其子进程、无运行错误。队列累计活动10.75024小时。预训练20480更新/8子集轮、两项完整编码、各三个完整validation-only候选均结束。当前没有后台训练继续，不能再沿用上午“正在第2个Cloth策略”的状态；没有自动重开候选或正式十seed。

Cloth lr.001/.0005/.0003分别完成74/213/163轮，验证选64/193/143轮，完整validation MRR×100分别29.4586873/30.1436390/30.0213526；最佳30.1436390比旧验证参考30.2534737低0.1098347pp。WN lr.05/.01/.005分别完成480/806/484轮，验证选380/706/384轮，完整validation Accuracy分别79.1034937/79.6308504/78.0158207%；最佳79.6308504比旧参考79.8286091低0.1977587pp。两项均未通过预声明至少+0.5pp改善门槛，故各新test=0，旧结果原样保留。这些新分数是validation，不能直接与论文test作差或称达标。

独立CPU重新核验六项全部保存artifact SHA、官方Cloth121323条validation IDs/排名/MRR和WN3034条validation pairs/labels/prediction/accuracy，全部轮次CSV/JSONL/TensorBoard、最终完整头/Adam/RNG/采样状态/历史/最佳轮/next_epoch及58份冻结源身份均通过；未新增GPU任务。证据artifacts/audit/cloth_wn_autonomous_results_verified_20261017.json。两项最佳候选的精确PNG已目视并绑定SHA；其余四图未实际查看，不伪标视觉通过，不修改完成账本的pending_visual_review字样。

当前可报告的完整test仍Cloth26.8109580%对论文28.88%，低2.0690420pp；WN77.9834078%对81.97%，低3.9865922pp，两项仍未达到严格绝对2pp目标。旧六项450结果、旧骨干、作者源文件和官方数据保持原样。本次扩大无标签池与固定学习率网格没有提供优于旧最佳的验证证据，不能作为“达标版本”交付；失败候选与所有日志保留，不依据test分数扩充搜索。以下为历史快照。

# Cloth/WN离线后台队列已实际续训（2026-10-17上海07:48核验）

用户最新明确授权代理直接运行，并希望断网后自动继续、自动尝试两项替代策略。已通过scripts/start_cloth_wn_offline.py启动独立后台入口scripts/run_08_cloth_wn_repair.py，启动器PID31668、实际控制器43968、GPU作业实际进程31152（PID仅为本次快照）；DETACHED_PROCESS和CREATE_BREAKAWAY_FROM_JOB请求获操作系统接受，启动工具退出后父子进程仍存活。不是定时自动化，原heartbeat保持暂停。无需再次点击入口。

新唯一运行账本artifacts/workflow_state/cloth_wn_autonomous_current.json=running_fresh_pretraining；原representation_coverage_current.json只读保留为父账本，其旧running不再用作当前进度。新run为runs/20261016T234711Z_coverage_poo_3ef021_s42_ff99dd，manifest父SHA核验为db5eb95bee2562d61653389f7db2442a4603911e23c8832c61563be922704089。已实际从13501更新至观察到13535/20480，subset_round6；不是重新初始化，也不是全图遍历。首批CUDA显存峰值2432.9MiB。新测试结果仍为0，不能称已达标。

本轮独立新增有限候选方案configs/cloth_wn_autonomous_20261017.json：Cloth固定lr.001/.0005/.0003三项；WN固定.05/.01/.005三项，完整验证、seed0、原官方图/划分/1024宽保持。先完成全部六个validation-only作业，再按验证最大值且相对原参考至少+0.5pp锁定保存权重，每项至多确认test一次；不根据test分数或论文差距扩充候选、不无限搜到2pp。剩余队列累计18小时活动预算，超限安全保存暂停，实际时间和2pp目标均不保证。细节见docs/cloth_wn_autonomous_20261017.md。旧KG模块中“LR retained”的历史通用描述不适用于本轮较小学习率候选，本轮以预声明profile及每run的evaluation.optimizer_lr为准。

36项入口/策略/恢复相关回归通过；新增两个实际CUDA头的连续3轮与第2轮模拟暂停后新run恢复，在独立CPU审计中模型、Adam、全部RNG、采样状态、历史、最佳权重/选轮和next_epoch逐项一致，PNG已目视并绑定SHA；fixture不是基准分数。首次验收中WN暂停函数的模块绑定未被fixture替换，造成模拟暂停未触发；仅修正验收hook后重做全部验收，旧失败输出原样保留。证据artifacts/audit/cloth_wn_offline_acceptance_verified_20261017.json和cloth_wn_offline_launch_verified_20261017.json。54份原冻结科学源与4份新队列源/入口身份核验，作者原文件/官方数据/旧结果无修改，无删除。

所有CLIP本地资产已核验、HF/Transformers使用离线环境，断网不影响此本地队列。控制器申请进程级防空闲睡眠，不修改全局电源设置；仍需接电、保持开机，关机/重启/手动睡眠/合盖策略可能中断。原子断点支持再用同一入口恢复，主动pause须明确--resume。其他任务维护暂停保留，不恢复Products/1471队列或十seed正式重复。日志artifacts/cloth_wn_autonomous_20261017/offline_console_20261016T234340237037Z.log与各新run日志持续本地保存；队列结束写同目录结果摘要.md，并保留未达标或待图像审查的真实状态。原六项450结果与旧骨干保持原样。以下为历史快照。

# Cloth/WN用户单入口准备与中断恢复核验（2026-10-17，上海）

本轮用户要求继续修复两项并给训练文件。新增scripts/run_08_cloth_wn_repair.py及独立adapter cloth_wn_repair_entry.py，PyCharm共享配置CAME_08_Cloth_WN_Repair；默认用户自己点击，代理未启动完整训练。基础Python39实际转交项目.venv-came39的完整--check-only已通过（exit0），根目录与scripts工作目录的--help均通过。独立交接审计见artifacts/audit/cloth_wn_training_entry_accepted_20261017.json。详细交接见docs/cloth_wn_training_entry_20261017.md。

实际发现上轮coverage控制器/训练子进程已退出，GPU全局锁及coverage锁均空闲，原ledger running为陈旧状态，未因该字样重启任务。原run20261016T021319Z_coverage_poo_3ef021_s42_6f4596最后日志step13672；一致原子断点last.pt仅提交13500（5个完成子集轮，正在第6），完整模型/Adam/Python、NumPy、Torch、CUDA RNG/固定池、顺序、游标有限数值与三图完整数据身份核验通过。父SHA db5eb95bee2562d61653389f7db2442a4603911e23c8832c61563be922704089，从13501继续，剩6980至20480，172条未提交日志更新需重算，不改旧实验状态/账本来伪称完成。

最新真实Torch2.1.2+cu118反向和DGL1.1.2+cu118消息传播通过，RTX3060 Laptop；既有完整作者模型/Adam/RNG恢复maxdiff0验收身份也重新核验。23项入口恢复拒绝/主动暂停竞争/GPU忙不重复/原固定池协议测试通过。额外项目512MiB+外部4GiB及每盘3GiB余量预算通过。原54份冻结科学源/计划/配置身份不变，新增入口未加入或篡改旧identity，无删除。

点击后串行原候选续训→Cloth全编码/完整validation→按门槛锁头test一次→WN同样流程；预声明val至少+0.5pp保留，不依据test搜候选或改门槛，不截全测试/降维/CPU回退。不自动执行两项正式十seed。主动pause必须明确--resume，且仅归档本流程pause，不清除Products或其他维护暂停。安全暂停/正常检查点进入新run带父SHA，保留logger/JSONL/CSV/TB/PNG。

旧确认结果仍Cloth26.810958（距28.88低2.069042pp）、WN77.983408（距81.97低3.986592pp），新coverage结果仍无；入口是有限修复验证，不能称已经达标或正式重复交付。原六项450正式结果、原编码器、作者文件、官方数据和所有历史产物保留；自动化保持暂停。以下为历史快照。

# 预训练覆盖单候选实际运行，CUDA/完整恢复验收通过（2026-10-16上海10:16）

有限诊断supervisor session39286、实际父12076/预训练子36816，representation_coverage_current.json=running_fresh_pretraining，run20261016T021319Z_coverage_poo_3ef021_s42_6f4596。实际三域新增无标签池32种子/1024原作者全模型forward/backward通过，最大allocated2583.77MiB、reserved2736MiB；不是全部子图容量保证。真实原作者模型中断恢复实测model/Adam最大差0、第二步loss完全相同，RNG与下一数据集/种子批/顺序/游标一致，证据artifacts/representation_coverage_20261016/joint_resume_acceptance.json，fixture仅验收不报基准分数。

目前仍为strict offline PPR准备阶段，Products32768/32768已完成，Books随后Cloth处理中，三域全池PPR完成后才允许GPU更新；不能把此阶段GPU低占用称CPU回退，也不能将running账本当已训练完成。新fresh8子集轮候选总20480更新，然后各一完整seed0验证，过冻结最强历史验证+0.5pp才锁权重test一次。新性能结果尚无，旧Cloth/WN test差距不变。

startup独立只读核验源/配置/作者原文件/旧编码器与旧三个PPR数据库SHA不变，见representation_coverage_startup_verified_20261016.json；20项CPU回归通过。已完成六项正式重复和Products提交前缀及维护pause均保护；正式重复未启动、无删除、无定时恢复。新方案开发已见过历史test，必须披露迭代历史，不能把本轮确认冒称全程未接触的盲测；重复seed也不能消除此选择偏差。以下为历史快照。

# Cloth/WN两项头修正均结束，转向单一预训练覆盖对照（客户端2026-10-16）

priority_cloth_wn_current.json=completed_bounded_priority_diagnostics_pending_manual_delivery_review，实际无Python训练进程；WN579轮/selected479，完整validation78.7079763%，比原低1.1206328pp；Cloth精修26轮/selected6，完整validation MRR30.2534737%，仅比先前结构候选高0.0660659pp。均未过预声明+0.5pp，new test0。audit_priority_rejected_candidates.py独立核验完整官方val pairs/labels或source/target/ranks、全部CSV/JSONL/TB、完整CP/RNG/最佳轮；两张精确PNG已目视、SHA绑定在priority_wn_rejected_independent_20261016.json与priority_cloth_rejected_independent_20261016.json。不能把validation当新test或称已修复；旧完整testCloth26.810958距28.88低2.069042pp、WN77.983408距81.97低3.986592pp。

用户继续明确授权改进Cloth/WN并查官方；新有限方案见docs/representation_coverage_20261016.md：仅固定无标签池×2，fresh seed42、8子集轮/20480更新，其他原已声明数值设置保留，不是warmstart追加8轮、不遍历全部节点。原池前缀逐项核验、旧SQLite只读backup全部逻辑行散列一致；旧模型/原作者/数据/全部历史结果不改不删。原上下文覆盖25%-38%只是待验证假设，不能宣称已定位根因。官方PMLR页面复查未见补全训练器链接，作者池大小/CLIP/PPR细节仍未知。

20项有关CPU回归通过；磁盘预留项目512MiB+外部4GiB通过。新入口scripts/run_representation_coverage_validation.py将在真实三域CUDA容量和完整Adam/RNG/采样恢复验收后才启动这个候选，之后Cloth/WN各一完整seed0验证，固定+0.5pp门槛对最强已有验证；通过才锁权重test一次。当前只是准备完成，尚无新性能结论，正式重复未启动。Products已提交124752前缀和维护pause保持；六项正式结果保护，定时仍暂停。以下为历史快照。

# WN新调度候选已审计拒绝，Cloth精修实际运行（客户端2026-10-16）

priority_cloth_wn_current.json=running_validate_cloth，无活动错误，supervisor session9783。WN真实全量候选run20261015T233259Z_priority_wn_667b4f_s0_39878e在579轮按100patience结束、验证选479轮，官方3034 validation accuracy78.70797626895%，比原79.82860909690低1.12063282795pp，未通过+0.5pp门槛，new test0，原完整test77.9834078距81.97低3.9865922pp保留。独立核对全部official validation pairs/labels/prediction、579轮CSV/JSONL/TensorBoard、最终完整CP579/next580及RNG/optimizer/scheduler、最佳轮与PNG目视，证据priority_WN_validation_rejected_verified_20261016.json。不能称WN已修复或发布达标版，纯头优化已有多次不改善，下一阶段如继续需审查表示/预训练依据，不能按低test不断搜参数。

Cloth当前新精修run从既有validation-best头出发，lr.0001，上限80完整train-positive轮、20patience，最近完成11轮，尚无新test；严格保留完整validation至少+0.5pp才锁权重test一次，不降低门槛来凑论文。现有确认test仍26.810958对28.88低2.069042pp。活动backend/profile/source保持冻结，所有六项正式重复/作者文件/原数据/旧产物保护。

新增一次性CPU依赖验收session75919：只等待本轮既有控制器37844退出，若账本完整结束再执行audit_priority_pair_results.py；失败/暂停则停止，不重训/重测，不是定时自动化。保存完整官方pred/rank及全部日志/头/父SHA核验后仍需代理实际查看精确PNG再seal，不能自动伪造visual passed。没有正式十seed训练启动。Products进程已退出，已核验124752游标前缀保留，维护pause不解除，不自动恢复Products。以下为历史快照。

# 优先Cloth/WN真实验收通过，WN正式完整验证运行（客户端2026-10-16）

独立新supervisor已实际启动session9783/父37844/子WN30928，ledger priority_cloth_wn_current.json=running_validate_wn。源码/配置冻结，当前run20261015T233259Z_priority_wn_667b4f_s0_39878e；最近读到完成525/2000轮、validation78.64205669%，仍在训练、尚无新test，不可与论文test混比或宣称达标。Cloth低LR精修随后串行。

修复后的真实Torch+DGL CUDA与两任务seed1恢复验收重新通过，报告priority_cloth_wn_CUDA_acceptance_20261016.json绑定backendSHA cf98adb65a24a0b93c735780945bcff30629705998136a6fc4edc497a1613ca1、profileSHA0f34fafc3ae81009ddad0ffb4213074296ee238aabbbf57934c992ab89695f9e。CPU独立逐项重核验全部RNG、采样generator、optimizer、scheduler、权重、beststate、history与next_epoch逐位一致，证据priority_cloth_wn_RNG_independent_verified_20261016.json。8项CPU检查通过，旧首次输入错误和验收/源码字节已归档，未隐去失败记录。

新增独立CPU审计入口scripts/audit_priority_pair_results.py，不改/导入活动训练源码：完整确认后独立核验官方val/test预测或排名、CSV/JSONL/TensorBoard、完整最终CP/RNG/scheduler、保存头/父SHA并生成pending视觉审计。必须代理实际查看原图且SHA相同后才能seal；未验收/未测试的版本不能作为达标交付。该入口不启动GPU/重测/正式重复。

Products原缓存游标124752前缀核验完整，实际无Products进程，本次优先两项期间不恢复；旧post监督pause保持，不能凭旧encoding账本自动启动。六个已完成正式重复/作者/原数据/全部旧结果均保护。Cloth当前已确认26.810958距论文低2.069042pp、WN77.983408低3.986592pp暂未新达标，正式十seed训练留用户。无定时自动化恢复。以下为历史快照。

# 用户优先Cloth/WN新有限诊断准备（客户端2026-10-16）

Products实际Python进程已全部退出，原encoding账本陈旧，退出原因未知；游标124752/2449029，已提交前缀SHA1bc94fccf46e93ce8f5ee53d7831e6b8ed578f0be4fd2972411b72400f9c9ec3重新核验一致，原缓存/源/账本不改。用户最新明确要求先完善Cloth/WN，新增post_sports_main_repairs_pause.json注明优先级，暂不恢复Products或旧队列；六项正式重复受保护。preflight优先计划/512MiB预算/全局GPU锁/作者SHA均通过。

新方案configs/priority_cloth_wn_20261016.json、docs/priority_cloth_wn_20261016.md，各1候选，先WN后Cloth。WN固定当前最佳图/cache/1024、头尾拼接线性头、CE、seed0初始lr.05/2000上限/100patience，仅加入validation CE ReduceLROnPlateau(.1/20/min.00005)。依据旧末100轮validation62.43-79.17%强烈震荡，非已确认根因。Cloth从验证选第97轮最佳头新建精修阶段，lr.0001/80上限/20patience，明确新Adam而非同一optimizer恢复，K4/图/全部官方划分保持。不降验证+0.5pp门槛、不用test选参数或seed。正式重复由用户训练，不先发布未验证达标文件。

首轮两任务seed1真实Torch+DGL CUDA和optimizer/scheduler/history/权重恢复对照通过，但WN正式输入接入时旧summary没有embedding_cache字段，训练开始前报KeyError；无基准run/训练/test产出。已在所有进程退出后修正summary/config双schema读取并拒绝路径冲突，两项实际完整缓存/官方数据身份均核验；8项CPU检查通过。另将纯工程CP频率设WN每25/Cloth每5轮，任何协作暂停/早停/最终轮必保存完整状态；日志/指标仍每轮，科学训练数学/参数不变，减少重复磁盘扫描。旧源码/账本/已通过CUDA报告逐字节归档于artifacts/audit/priority_input_resolver_migration_20261016，source/profile身份迁移独立审计，无活动热改。新代码将重新进行完整实际CUDA恢复验收后才开始基准。

专用入口scripts/run_priority_cloth_wn_validation.py、ledger priority_cloth_wn_current.json。原Cloth完整test26.810958距28.88低2.069042pp、WN77.983408距81.97低3.986592pp仍未达严格2pp；这只是新诊断准备，不是已修复。Products保持断点，Books协议未知，旧正式结果和作者文件/数据不删除，定时仍PAUSED。以下为历史快照。

# Cloth新测试26.81仍未达严格2pp；WN候选拒绝；Products编码遇内存压力（2026-10-15）

Cloth结构候选完整107轮，验证选97轮，validation0.301874078058比参考提高0.723548pp，锁权重后完整97456官方测试/150负例一次test=26.8109580237%，论文28.88，差-2.0690419763pp，比上版26.1155809836提高0.695377pp；离严格2pp门槛尚差0.0690419763pp，不能四舍五入称达标。全部val/test排名、CSV/JSONL/TensorBoard、选头/父SHA已独立核验。hard_lp_curves.png已目视并绑定SHA823f1d716649c2a136e71c2e1a0d5c01c0861a9e7ef8448282da6638e09c322b，补充独立视觉审计cloth_predictor_confirmation_visual_verified_20261015.json；原活动账本与完成审计不改。

WN同一strong-Wolfe候选从261继续至449，selected349，validation78.5102175%，比固定参考低1.31839156pp，按预声明门槛拒绝，新test0；保留既有完整测试77.9834078距81.97低3.9865922pp，未新达标，不重测/挑seed。监督已进入running_full_Products_single_seed。

Products实际进程11288/父33940运行，run20261015T062548Z_products_ful_2503dc_s0_8b9e61。完整2449029节点编码游标约124672（5.09%）仍增长，日志累计约22242.8s/吞吐5.61节点每秒，最近16节点batch约45秒，吞吐不稳定不能保证全程时长。实测GPU使用5926/6144MiB、利用率17%，系统RAM只余317/16069MiB，进程private18,821,971,968 bytes、working set846,508,032 bytes，存在明显主存压力。不可据GPU单次低利用率称CPU回退；CUDA任务仍在运行，实际无失败。已保存监测资源快照。容量小样本验收不能覆盖长程内存压力，后续如优化须先审计/新方案/精确数值等价，禁止热改活动源码、暂停/停止现有任务或第二GPU作业。当前未中断任何进程，Products无test结果。Books协议仍未知，保持原结果。

Sports十seed/前五项正式重复继续保护。四余项尚无新增严格2pp达标版本，不能交付伪称达标文件；正式重复留用户。以下为历史快照。

# Sports十seed完整核验结束，Cloth结构诊断实际运行（2026-10-15）

用户已完成Sports正式重复并授权继续剩下四项单seed实测、正式重复由用户自行训练。Sports十seed官方完整val/test IDs/ranks/MRR/Hits/反向子集、每seed日志CSV/JSONL/TensorBoard、验证选轮和保存头/父SHA重新独立核验；mean35.7629591932%，sampleSD0.0786681453pp，论文37.68，差-1.9170408068pp，均值达绝对2pp目标。9/10单seed在2pp内，seed9=35.5997290201差-2.0802709799pp，原样保留无丢弃。不是所有seed均达标，也不是十次独立预训练。新增最终汇总artifacts/final_results/sports_ten_seeds/verified_results.json、results.csv、paper_comparison.png和validation_histories.png；两张汇总图已目视。原Sports账本/实验/清单不改，入口再次无须重训。

后续监督22748与实际Cloth GPU进程29420已核实，post_sports_main_repairs_current.json=running_validate_cloth_predictor。真实Torch+DGL CUDA、四候选难负例选择、seed1完整optimizer/RNG暂停恢复逐位一致均通过，证据cloth_predictor_CUDA_acceptance_20261015.json。当前Cloth run20261015T044740Z_cloth_head_s_878f21_s0_8f49d4已完成9/300轮（可按10patience早停），validation0.28257763；本候选test尚无，不可用validation与论文test比宣称达标。只有完整validation达到0.299638596907才锁头做一次完整test。活动源码/profile冻结，未加第二GPU任务。

Books新增固定官方源码MRR一致性审查：113个合成queries、每个150负例，包含全平分/最好/最差rank，当前MRR与官方torch evaluator最大差3.20e-9，Hits完全一致。固定loader实际读取lp-edge-split.pt，README的random/hard字样不是该版本的文件选择开关；在线官方发布目录仍列此文件，不存在已确认的替换划分证据。books_official_evaluator_rechecked_20261015.json记录来源/归档SHA。不能用换指标/造负例/降分凑10.42；作者协议仍未确定，Books尚未修复。

WN原261步/next262续跑与Products完整2449029节点/2213091test由既有冻结入口在Cloth结束后串行执行。六项正式重复完成（430NC+10FB+10Sports=450结果），加预训练及WN/Cloth/Books诊断共454/1471逻辑位置；剩1017逻辑位置，旧strict427不改。不能据此说其余四项已经测试达标。定时保持PAUSED，没有删除数据/模型/历史结果/作者文件。以下为历史快照。

# 最后冻结审查通过，Sports最后一个seed仍在训练（2026-10-15）

用户要求不打断当前任务，先作最后严谨修正。当前Sports已确认9/10结果，mean35.7810958791%，sampleSD0.0571112401pp，论文37.68，差-1.8989041209pp；最后seed9训练中，完整十seed仍未完成。未改Sports或排队监督的任何冻结源码/profile，没有停止进程或新增GPU任务。新增独立CPU只读审计scripts/audit_final_main_repair_plan.py：作者原文件/Sports和排队source全SHA、Cloth仅预测头结构变化、WN原父SHA/预算、Products完整节点/维度/测试量、validation筛选/test一次约束均通过；证据final_main_repair_plan_verified_20261015.json。相关13项CPU回归通过。不是新性能验收，不能保证四项最终2pp或称已达标。

监督继续waiting_for_user_Sports_ten_seeds：用户Sports十seed完成/GPU释放后按既定单seed方案串行Cloth预测头CUDA恢复验收与完整validation、WN同原断点续跑、Products完整评估。Books仍待作者评测协议可比性证据，不人为降分。当前逻辑位置453/1471（Sports新seed1-8八结果），全部Sports完成才454/1471；旧strict账本不变，正式其余重复由用户。定时仍PAUSED；本监督是用户授权连续入口。以下为历史快照。

# 四项主任务优先修复准备：客户端日期2026-10-15，Asia/Shanghai

Sports用户正式训练仍正常：已确认seed0/1两结果，均值35.7583973536%、sampleSD0.0889155489pp，距论文37.68低1.9216026464pp；全十seed未完成，当前seed2训练，不能提前宣称十seed达标。活动Sports所有冻结源/清单SHA重新核验一致，原作者三文件SHA一致，无删除。

Cloth26.1155809836距28.88低2.7644190164pp；WN77.9834077856距81.97低3.9865922144pp；Books23.7908054318高于10.42约13.3708054318pp、协议可比性未解决；Products无完整test。严格方案docs/post_sports_main_repairs.md和configs/post_sports_main_repairs_20261015.json：等Sports全部十seed/GPU空闲后，串行一个Cloth匹配官方基准预测头结构候选（先实际CUDA/seed1断点逐位验收）、WN同一冻结solver从261/next262完整断点继续、Products完整精确推理与全测试。每次仅完整validation至少+0.5pp才锁权重test一次，无test筛选。Books没有足够作者协议证据，继续可比性审查，不人为降低高分。

六项新CPU针对性检查通过；WN完整config/optimizer/RNG/history/有限数值和父SHA30e81df4dfb649953979ca1151674a1ef92afb46806870b986aca2341e3b775d核验，Products新增9.34GiB及512MiB头预算通过守卫。CPU审计post_sports_main_repairs_preflight_20261015.json。正式重复由用户运行，监督不启动正式十seed/旧1471/基线/fewshot/消融，不解除历史优先暂停，不恢复定时自动化。当前逻辑位置446/1471（Sports新增seed1一结果），Sports全十完成才454/1471；旧strict427账本不改。新入口准备，不是达标交付。以下为历史快照。

# Cloth本轮完成但未达2pp，Sports用户训练可用：客户端日期2026-10-15，Asia/Shanghai

Cloth K4候选按原10patience在完整245轮结束，validation选第235轮MRR0.294638596907，比旧0.288640794605提高0.5997802302pp，通过预声明门槛后锁头test一次。new confirmation run20261014T233122Z_protocol_con_2ad33a_s0_b5f8a6，全部97456官方queries/150固定负候选，MRR×10026.1155809836，比旧25.5106929383提高0.6048880452pp，距论文28.88差-2.7644190164pp，仍未达绝对2pp目标；17360反向重叠单列，不重叠MRR×10025.8835470487。全部validation/test ID/rank/MRR/Hits/反向mask、245完整轮CSV/JSONL/TensorBoard、最优头/父SHA独立核验，PNG已目视；证据artifacts/audit/two_lp_handoff_cloth-copurchase_20261014.json。不能将validation29.46与论文test28.88直接比较或宣称达标，不发布Cloth达标文件。

two_lp_hard_negative_current.json=completed_bounded_hard_negative_diagnostics、handoff=pending_visual_review仅Sports eligible；实际无诊断Python进程，GPU锁释放。本轮不再重跑/恢复已结束候选，正式重复仍由用户。Sports已独立不可变第07号交付scripts/run_07_sports_repeats.py，seed0测试35.6955245660距37.68低1.9844754340pp；可以用户点击补seed1-9，未来均值不保证。无需等待未达标Cloth；不改该交付清单和已达标五项完整重复。

剩余Cloth已做source/train-validation结构审查：固定官方MMGraph归档SHA重新核验，lp/models.py127-134 predictor是两个Linear/LayerNorm/ReLU隐藏层；现Cloth是一个Linear/ReLU隐藏层，未匹配预测头结构。该差别不是已证明根因、也不是CAME作者确认。下一至多一个保持1024与全划分、同图/encoder/K4/seed0/optimizer/budget的匹配头候选须先实际CUDA/完整validation验收，门槛至少0.299638596907；不依据test选参数或seed、不追测、不缩集。当前只记录方案与证据cloth_predictor_structure_review_20261015.json，新GPU作业0，保持用户Sports训练资源可用。WN/Books/Products仍未完成交付，整篇未复现成功。

更新协议逻辑位置445/1471不变，strict旧427不回填，Sports新版本和Cloth新版本仍同一seed0位置。无删除、无源/原数据/作者文件改动、定时保持PAUSED。以下为历史快照。
# 外部中断恢复成功与第07项独立交付：2026-10-14T20:39:30.2805040+08:00，Asia/Shanghai

用户明确授权继续不小心中断的诊断，完整训练由用户自己执行。实际无Python诊断进程/GPU锁空闲后核验冻结src/profile/候选/官方数据/完整缓存/encoder/19轮完整head与optimizer/RNG/generator/history，parentSHA eff72b7c5e1e6b037d711d5b94fac4531070a15a36db59d0e2f4e68df803a957；审计two_lp_external_interruption_20261014T122905Z.json。只修复无活动父控制器重复ready_tasks计数的一条membership条件，原账本字节归档并审计old/new controllerSHA；训练候选源码、profile与科学协议未改，见two_lp_resume_controller_dedup_migration_20261014.json。

原保存入口已恢复，session51723；new run20261014T123209Z_hard_lp_clot_42d368_s0_fa6c27父SHA核对一致、19轮历史逐项一致，第20轮继续，当前28/300轮validation MRR=0.261289913097793，实际PID24276。没有从头训练Cloth，没有重训或重测已完成Sports。最长300轮/10patience，完整验证至少+0.5pp门槛和test一次保持。新Clothtest尚无，旧确认25.5106929383距28.88差-3.3693070617pp，不拿本次validation宣布达标。

Sports第07号已单独交付scripts/run_07_sports_repeats.py；固定不可变清单verified_job07_sports_delivery.json，只复用已完整核验seed0=35.6955245660，距37.68差-1.9844754340pp，补seed1-9。默认Python39自动venv的dry-run和2项防训练负例混用检查通过，正式GPU作业0。Cloth占GPU时入口提示GPU_BUSY不抢占；结束后用户点击。第07号是主任务交付顺序，不是1471索引。用户指南docs/job_07_sports_training.md，验收job07_sports_delivery_acceptance_20261014.json；未来十seed均值尚未测，不能保证在2pp内。通用入口支持独立delivery/ledger，防止未来Cloth交付改变Sports恢复身份；无活动handoff源码身份迁移另存numbered_lp_delivery_identity_migration_20261014.json。

交付审计等待程序已恢复session54470/实际PID10756，只在当前有限诊断结束/GPU释放后审计，不启动正式重复。未触碰五项完整重复、作者文件、原数据、旧WN优先暂停。逻辑位置445/1471不变，严格旧账本427；定时PAUSED保持。以下为历史快照。
# Sports单seed进入2pp目标，Cloth继续：2026-10-14T18:31:11.0995510+08:00，Asia/Shanghai

新Sports K4难负例候选完整60轮，validation最大第53轮MRR0.387925990879，较固定参考0.376942663890提高1.0983326989pp，通过预声明验证门槛后锁头test一次。完整37413官方test/150固定负候选MRR×10035.6955245660，论文37.68，差-1.9844754340pp，单seed达到绝对2pp目标；较上一34.9196200965提高0.7759044695pp。6621反向重叠单列，不重叠MRR×10035.0967773144。独立复核45557全部validation与37413test IDs/ranks/MRR/Hits、反向mask、CSV/JSONL/TensorBoard60轮、保存头/父SHA和PNG目视；证据artifacts/audit/sports_hard_full_artifacts_20261014.json。该结果接近2pp边界，不能保证未来十seed均值仍达标，更不是完整论文复现声明。禁止为了余量依据test加调参、换seed或缩测试集。

当前实际Cloth hard validation-only run20261014T101714Z_hard_lp_clot_42d368_s0_67f8ff，PID12132，完成12/300轮，validation MRR=0.250178926377887，没有本候选新test。冻结原架构/1024编码/官方图/全划分/K4/lr/seed0，按完整validation至少+0.5pp筛选；全程单GPU。最新已确认Cloth仍25.5106929383距28.88低3.3693070617pp。

two_lp_hard_negative_current.json=running，Sports ready_tasks为单seed测试门槛通过，不等于用户正式重复已经完成；handoff等待Cloth有限诊断结束再做交付审计，不修改活动源或启动正式十seed。当前五项完整重复达标加Sports一单seed达标=6主任务在2pp内；WN/Cloth/Books3项未达可比目标、Products1项未完整测。更新协议逻辑位置445/1471不变，旧strict427仍保留；一个seed0的新版本不增加作业计数。用户原定时PAUSED偏好保持。以下为历史快照。
# Cloth收敛完整测试与难负例阶段启动：2026-10-14T17:52:37.0151584+08:00，Asia/Shanghai

Cloth300轮验证完成，验证选第297轮MRR0.288640794605，保存头完整97456官方test/150固定负候选确认MRR×10025.5106929383，比原24.2274958277提高1.2831971106pp，距论文28.88低3.3693070617pp，仍未达目标；17360反向重叠单列，不重叠MRR×10025.3141605063。独立重算全部validation/test IDs/ranks/MRR/Hits/反向mask，核对CSV/JSONL/TB300完整轮、选择头和父SHA；PNG已目视，证据artifacts/audit/cloth_convergence_full_artifacts_20261014.json。Sports已确认34.9196200965，差-2.7603799035pp。前段two_lp_priority_current.json=completed_two_lp_bounded_diagnostics，无达标新增任务。

下一固定K4难负例候选实际开始CUDA/seed1暂停恢复验收，实际PID30224/session8838、run20261014T095133Z_hard_lp_resu_1c5adc_s1_c36595；小随机fixture不是benchmark，不能把其分数报告为Sports/Cloth。只有实际Torch+DGL CUDA、正确难负例/非边过滤、完整optimizer/RNG恢复逐位一致通过后才启动Sports再Cloth正式单seed诊断。handoff进程继续等其结束，不启动正式重复；活动源码/profile仍冻结。未依据test修改K/lr/seed或裁测试数据。

范围仍445/1471更新协议逻辑位置，重做旧seed0不增加；五项已达标固定重复受保护。两项尚无可交付达标版，WN/Books/Products状态不变；定时自动化仍PAUSED。以下为历史快照。
# 两项LP继续实测与交付验收：2026-10-14T17:38:37.5295782+08:00，Asia/Shanghai

用户要求继续把Sports/Cloth向论文靠近，实际诊断持续运行。Sports收敛60轮完整确认MRR×10034.9196200965，论文37.68，差-2.7603799035pp，未达绝对2pp；45557完整validation和37413完整test/150固定负候选、反向子集、CSV/JSONL/TensorBoard源数值、选择头/父SHA独立核验，PNG已目视，证据artifacts/audit/sports_convergence_full_artifacts_20261014.json。

当前Cloth run20261014T083624Z_convergence__c821b8_s0_2629b7完成270/300轮，validation MRR=0.287180141579795，没有新增test，不把验证分数与论文test当作同一数值。实际训练PID12328；two_lp_priority_current.json=running。下一预先声明的固定K4 train-only难负例候选two_lp_hard_negative_current.json=waiting_for_two_lp_convergence_completion，实际等待PID11196，session8838；不得修改其冻结源码/profile、启动第二个GPU、据test搜索K/lr/seed。原WN优先暂停文件保留，Products不抢占这两项。

新增独立交付基础设施scripts/audit_two_lp_confirmation.py、run_verified_two_lp_repeats.py和prepare_two_lp_handoff.py，不改src/作者文件/现有活动候选。正式重复入口必须匹配交付负例规则，拒绝hard4与uniform1混用，固定复用seed0补1-9，记录全部seed。四项定向CPU检查通过，dry-run明确NOT_READY/GPU jobs0；通用uniform入口新增seed1完整断点一致验收，尚未实际运行GPU验收，不能宣称通过。串行handoff session16675实际等待PID10128，只在有限诊断完整结束/GPU释放后做必要验收与独立重算；最多产出pending_delivery_visual_review.json，必须代理目视PNG并重新绑定SHA，才允许verified_delivery.json，绝不自动声称视觉验收通过。不启动用户正式重复，预计两项18个新下游seed头额外预留1GiB，原30GiB/40GiB/3GiB守卫保持。

仍五项固定重复结果受保护，更新协议去重覆盖445/1471逻辑位置，严格旧账本427不变；重跑同seed0不增加数量。WN/Books尚未达可比目标、Products未完整测试；没有整个论文成功声明。作者三原文件SHA一致，未删除文件。定时监测保持PAUSED，本次为已授权连续实验。以下为历史快照。
# 第05项十seed完成核验，用户授权剩余诊断恢复：2026-10-14T15:58:38.5651015+08:00，Asia/Shanghai

FB15K237正式固定十下游seed全部完成并独立核验：mean93.3895240887%，sample SD0.0878738023pp，论文94.59，差-1.2004759113pp；最低93.1887032151/最高93.4818723737，十个seed均在绝对2pp以内。seed0复用不重训，seed1-9均保留，不选seed、不丢低分；同一个预训练模型，不宣称十次独立预训练或整篇成功。所有官方20466 test pairs/labels/prediction accuracy、每seed validation最优epoch/CSV/JSONL/TB、selection锁/头SHA/parentSHA已核验，汇总PNG已目视核对。证据artifacts/audit/FB15K237_ten_seed_results_verified_20261014.json，图artifacts/validated_remaining_repeats/FB15K237_ten_seed_comparison.png。第05号入口再次运行只提示完成，不重新训练。

旧1471清单严谨核算：原旧账本strict completed仍427（1预训练+426原协议eval），保持历史不回填。修正四NC430+FB10=440个已完成重复结果；另有1预训练和WN/Sports/Cloth/Books各1完整但不达目标的诊断。按(task,dataset,representation,pretrain seed,split,downstream seed)与旧清单去重映射，共445/1471个逻辑位置覆盖、1026未覆盖；不是445个原协议完全等价作业，不能把旧427再相加或把val训练/保存头test拆成两个科学结果。逐项crosswalk在artifacts/audit/legacy_1471_logical_coverage_20261014.json，明确protocol change。尚未覆盖CLIP全基线、fewshot、九消融和其预训练等，不以五主任务完成冒称全部完成。

用户最新要求继续修复余项并交付真正达标的版本，优先暂停约束在FB退出后已解除。新串行监督scripts/resume_remaining_diagnostics_after_task_05.py已启动，session62728、实际30648，先等FB账本complete10与控制器/GPU锁释放，再核验原暂停138步/next139完整checkpoint/SHA、所有冻结source/profile/author身份，归档该优先pause并保存审计artifacts/audit/task05_resume/20261014T075300Z_resume_audit.json，只删除已归档的特定维护pause。未重训任何FB/四NC。新WNresume run20261014T075324Z_wn_wolfe_con_4e29af_s0_434d61，parentSHA c5c981d031a9da09d4074e2d0b4bbb9025a4b46f907a5292415cb0ce3baf64b8，继续原同一solver预算/编码/官方validation，目前165步，未新增test。收敛队列WN/Sports/Cloth各1候选+0.5pp完整validation门槛，锁saved权重后才test一次，弱候选拒绝。下一步只有当前GPU诊断完全退出后才Products完整单seed；该入口保持2449029节点/1024FP32/2213091官方test/前缀SHA与同cursor OOM减批重算，无CPU计算回退、无裁test。正式余项十seed仍待用户自己点击。

最新已确认但未达2pp：WN77.983408距81.97低3.986592pp；Sports33.559245距37.68低4.120755pp；Cloth24.227496距28.88低4.652504pp；Books23.790805比10.42高13.370805pp，仍按协议可比性审查，不人工降分/换指标/捏造负候选。Products未完整测试。当前无余项新增达标版可交付，必须完整核验后才新增可训练入口。2pp是目标不能事前保证，来源未知必须保留。监测原PAUSED设置保留，监督GPU串行流程会自行继续。

以下为历史快照。
# 用户优先接手主任务05：2026-10-14T13:34:23.7772252+08:00，Asia/Shanghai

用户最新要求立即提供已诊断达标、未做正式重复的独立训练文件，并记录主任务编号。入口scripts/run_05_fb15k237_repeats.py，优先顺序主任务05=FB15K237，接在已完成的四NC之后，不是旧1471队列索引。只复用已核验seed0完整test93.481872%，论文94.59，差-1.108128pp；固定原验证选定train_reverse/raw头尾拼接线性头/Adam .05/2000上限/100patience/全官方划分/同一冻结encoder，用户启动后补seed1-9各一次validation选最优保存头及完整test确认，不挑seed不删低分。十个下游初始化seed不等于十次独立预训练。其他仍未达标或未测任务不混入该入口；前四NC不重训。单seed达标不保证未来十seed均值达标。

用户基础Python39运行该编号入口--dry-run成功，显式转交项目venv，GPU正式作业启动0；13项新定向CPU检查通过。配置configs/validated_remaining_tasks_20261014.json；编号记录artifacts/validated_remaining_repeats/job_05_ready.json；账本validated_remaining_repeats_current.json=ready_for_user_run。正式完整训练由用户点击，不由代理启动。

为满足用户立即优先训练需求，当前convergence候选通过暂停文件安全保存后退出：WN strong-Wolfe完成138步/next139，完整optimizer/RNG/history和SHA核验，GPU诊断已释放。convergence_repair_current.json=paused，session10717退出1由父控制器对SAFE_PAUSE产生，不是训练数值故障；不能据此自动恢复抢占用户GPU。维护pause convergence_repair_pause.json保留reason=user_priority_validated_FB15K237_repeats，用户FB重复结束前不清除、不启动后续Sports/Cloth/Products诊断。审计job_05_handoff_safe_pause_20261014.json；作者三原文件SHA仍一致。

前轮有限图文修复已结束：WN新完整test77.983408%，差-3.986592pp，全部3134pairs/labels/accuracy和选择锁核验；Sports33.559245差-4.120755pp；Cloth24.227496差-4.652504pp；仍未达2pp。Books23.790805高于10.42，协议差异未知，不按test降分或换指标凑目标。Products容量及1024维全邻居数值校准通过（batch16，最大差1.41e-6，CPU峰值7.45GB），完整单seed尚未启动；新scripts/run_products_single_validation.py已补前缀校验/逐行FP32写入/同未保存cursor减批重试，7项CPU IO与游标/损坏拒绝测试通过，仍须后续完整实测，不抢占用户训练。

监测保持原PAUSED偏好；正常诊断不得自动恢复。以下为历史快照。
# Cloth新测试确认与WN阶段切换：2026-10-14T13:00:41.0215933+08:00，Asia/Shanghai

Cloth双向去重候选完整100轮，validation选第100轮MRR0.274225642445（有向control0.254386788455），通过预声明+0.5pp门槛。锁定保存权重后确认run20261014T045635Z_protocol_con_2ad33a_s0_4c3c21完整97456官方test/150固定负例一次test，MRR×10024.2274958277，较旧22.4420807409提高1.7854150868pp，距论文28.88低4.6525041723pp，仍未达标。17360反向重叠单列，80096不重叠MRR×10023.7386722206。source/target/rank、reverse mask独立重算、CSV/JSONL、TB事件、验证最优轮、选择锁和权重/cache SHA、所有活动源码SHA核验；训练PNG目视与源数值核对。证据artifacts/audit/cloth_graph_protocol_confirmation_verified_20261014.json。不据此改test或热追加轮。

WN新OFA前缀文本及全实体冻结编码已完成，当前validate_WN18RR，runE:\danzi\A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs\runs\20261014T045920Z_kg_affine_WN_fce445_s0_cb5aeb，最新指标轮221，仅validation，不宣称新测试达标。实际GPU子进程31856已核对。Sports已核验33.559245，仍低论文4.120755pp；Products继续等待本串行流程GPU释放后校准。所有五项达标结果保留。计数仍5项差2pp内、3项偏低、1项Books偏高协议审查、1项Products未测。

以下为历史快照。
# 修复实时核对：2026-10-14T12:22:40.6093518+08:00，Asia/Shanghai

Sports train-only 双向去重候选已完成验证选权重及完整官方测试，MRR×100=33.5592451851，比上一个28.9835566756提高4.5756885095pp，距论文37.68仍低4.1207548149pp，尚未达2pp目标。全部37413官方source/target/rank、日志与选择锁/父权重SHA已核验，见artifacts/audit/sports_graph_protocol_confirmation_verified_20261014.json。

Cloth同初始权重有向control完整100轮完成，validation MRR=0.2543867884553178，选第99轮，不做test；新的train-only双向去重编码也已完成，当前run20261014T042118Z_protocol_lp__f9bd3b_s0_bbec2c正在validation-only头训练，最新完成4轮。只有超过旧参考及同初始化control至少0.5pp才锁权重做一次完整test。下一阶段WN固定OFA文本前缀候选尚未开始。实际主进程5372、活动GPU子进程28236已核对，未出现失败。

Products资源/数值校准已通过独立scripts/run_products_queued_calibration.py排队，session27317、实际等待进程18220；products_capacity_current.json=waiting_for_protocol_repair_GPU_release。只有当前主流程结束、GPU锁释放后才开始实际Torch+DGL CUDA/显存/内存/磁盘/全邻居数值校准，尚无Products测试分数、尚不是完整单seed。当前只有一个GPU任务。5项已达标结果、作者原文件和全部历史实验保持不变。

只读CPU PPR诊断已完成，无新增编码器、无validation/test反馈、未改旧PPR数据库。对固定未标记seed更严格epsilon改变部分PPR邻域与残差，证据artifacts/audit/ppr_tolerance_structure_diagnostic_20261014.json；不能据此声称已经找到根因或提分，后续预训练分支须另外冻结预算与验证协议，活动源码/profile不热改。

Books已核验23.790805，高于论文10.42约13.37pp，仍为协议可比性审查范围，不降分凑目标。当前计数仍5项在2pp内、3项偏低、1项偏高需协议审查、1项Products未测；四项绝对差超目标均保留修复/审查范围。以下为历史快照。
# 全部超目标任务开始修复：2026-10-14T11:39:55.2224030+08:00，Asia/Shanghai

用户最新授权立即修复所有距目标超过2pp的作业，撤回之前延后要求。范围WN/Sports/Cloth/Books协议审查，Products完整评估后判断；五项已达标前四NC430及FB保持原样。configs/protocol_repair_20261014.json在候选前声明，新增独立模块protocol_repair.py、all_unmet_repair_pipeline.py及scripts/run_all_unmet_repair.py，未改旧生产源码/作者文件/旧配置/原始数据/历史run。

111全回归通过；实际Torch+DGL CUDA、作者1024维全图/精确两跳推理差5.96e-8、两头损失有限梯度、完整头checkpoint/RNG/history暂停恢复逐位一致，父SHA核验。验收artifacts/audit/protocol_repair_acceptance_20261014.json；预留额外1.5GB缓存，既有11.42GB外部cache、C空闲56.8GB/E27.3GB且30GiB/40GiB守卫通过，没有删除任何缓存/结果。训练图canonical Sports603696/Cloth1622678边，只来自原train+reverse去重/无原self；10万train-only负采样诊断分别17/8个反向train冲突，保留原负采样不将第二变量混入图试验。

当前scripts/run_all_unmet_repair.py已启动，session49415，ledger all_unmet_repair_current.json=running。Sports新图50250全节点编码52.5秒完成，当前新run20261014T033233Z_protocol_lp__c33d12_s0_eb37c6验证头，实际进程5488及RTX3060只有一个GPU作业；最新validation MRR0.363751（旧参考0.320973），尚无新test，不以验证对论文test宣布达标。Sports仅一graph候选；Cloth先同recipe/同初始seed0有向validation control，再一graph候选，解决旧头RNG经过encoder构造导致不匹配，必须胜过两reference+0.5pp才确认；WN仅一固定OFA实体前缀并沿用旧selected图/affine/Adam/线性头。选最优仅validation，锁checkpoint/summary/cache状态SHA后一次完整test，弱候选复用原test。活动后源码/profile全部冻结，禁止热改、第二GPU或重新启动旧已完成KG/LP/修复入口。

Books没有被排除：官方2024年9/10月、2025年2/5月划分SHA均与本地一致；所有570464val/1710677test每行150负例，含自身正target的行分别122/442、含source151/393、非法ID0；发布数组保留。核验单次test23.790805高于论文10.42，不能用换metric/故意降模型或捏造1000负例来凑接近。作者LP头未公开仍是协议未知，需要真实来源而非按test反推。审计lp_historical_split_metadata_20261014.json和books_lp_candidate_integrity_review_20261014.json。图文候选结束仍需Products容量/数值校准和单seed，若有限screen不能全部修好必须如实记录，不宣布整篇复现成功。

监测提示已更新当前stage与最新授权；自动化现有PAUSED设置保留，不自行改变用户监测偏好。当前GPU串行入口独立运行会自行继续候选，不要再点重复实例。

以下为历史快照。
# Goodreads-LP完成与独立核验：2026-10-14T11:06:07.8959487+08:00，Asia/Shanghai

Books-LP单seed完整完成；LP ledger=completed_three_lp_diagnostics，session80479已退出0，无Python训练进程。55轮训练、validation选第45轮，完整1710677官方test/每查询150固定负候选一次test，MRR×100=23.790805（论文10.42，+13.370805pp，相对+128.31867%）；排除108818反向重叠后的1601859查询MRR×100=22.326093。该结果明显高于论文，不能按“偏低未达标”处理，也不能直接宣布论文复现成功；本地发布文件及指标已经核验，论文原始负候选/划分版本和未公开头仍需进一步一致性审查，不人为降低分数或重采负例凑接近。

保存source/target/rank、反向mask逐项等于官方；MRR/Hits及非重叠子集独立重算一致，raw split SHA36e3da17b5a6bd711721e5f00061bf725f9603acf3f1ec410a4895a5eadd6b94，55轮CSV/JSONL/TB、best/last权重/selected45/next56/stale10/RNG/optimizer/采样历史、父encoder SHA一致，PNG目视核验。证据artifacts/audit/books_lp_single_seed_verified_20261014.json。旧ledger error=exit75是已归档维护暂停的历史字段，不是本次失败；完成记录不为清理字段而覆盖。

十主任务现9项已有测试，1项未测Products：5项绝对差2pp内；3项偏低WN/Sports/Cloth仍需修复；1项Books高于论文13.37pp需协议一致性审查。若按绝对接近目标计，共4项超2pp，但不把高分当性能不足。当前无GPU诊断活动，Products尚待容量/完整全邻居数值校准，不能宣称后六项全部完成。用户最新要求新修复等当前这几项完成后实施，docs/deferred_unmet_repair_plan.md仍deferred；本轮只核验完成结果，不启动新GPU/正式重复/热改源码。五项已达标、作者文件及所有历史保留。

以下为历史快照。
# 官方资料诊断与用户延后实施要求：2026-10-14T10:56:00.9655882+08:00

用户最新要求现在查官方资料与构思，等当前几项完成后再实施新修复；不是暂停现有Books诊断。方案docs/deferred_unmet_repair_plan.md为review-only/deferred，不是可运行配置；新增GPU=0、活动源码/profile/数据/cache/作者原文件均未修改，并再次核验活动source/profile SHA一致。当前Books最新日志55轮，仍无已核验最终test，不据此改变计数。

PMLR正式19页PDF与本地作者文件SHA完全一致，排除版本不一致；出版页/PDF未找到完整训练代码链接。WN官方实体/关系映射及86835/3034/3134全部train/valid/test三元组核验一致；固定OFA实体文本存在前缀差异，CAME是否采用未知，不能叫确定bug。有限后续路线先LP train-only双向去重验证，WN文本语义审查及最多一个确切上游前缀候选，保持拼接线性头；实际验收/训练须等当前几项结束。若不足再基于train/validation设计新预训练分支，不自动追加轮或挑seed。

只读PPR v2邻域并集Products681240/2449029（27.82%）、Books157092/636502（24.68%）、Cloth48010/125839（38.15%）；不能将seed池比例当全部训练上下文，也不能因coverage推导必然提分。10240历史更新零模态记录为0、>10000梯度为0，各图裁剪比例有限，不能归因全面梯度爆炸。审计came_official_pdf_comparison_20261014.json、wn_text_protocol_review_20261014.json、pretraining_gradient_review_20261014.json、pretraining_context_coverage_review_20261014.json。监测提示已保存最新延后约束，不得先实施后通知。

以下为历史快照。
# 只读修复审查：2026-10-14T10:40:30.8506611+08:00，Asia/Shanghai

用户询问能否修复未达标项且不影响当前进程；已明确保护Books活动任务，未启动新GPU、未改任何生产源码/profile/cache/原文件，活动ledger全source/profile及作者SHA核验一致。Books最新完成45轮、最佳validation MRR0.238452，未test。保存artifacts/audit/read_only_unmet_protocol_review_20261014.json。

找到LP待检验差异：固定MMGraph commit0769fa670c69a6dc0756a2d7fc697b980e494775的lp/lp_dataset.py38-41只取train边，lp/main.py247-249再调用to_bidirected_with_reverse_mapping，lp/utils.py14-33增加反向并去重；当前CAME LP推理为原始有向train-only。有上游依据，不是盲目加反向，但CAME作者loader未发布，不能当已确认bug或达标承诺。GPU空闲后候选预案为Sports/Cloth各最多一个train-only双向去重视图，保留冻结encoder和各任务此前validation选定头配方，不扩大头/seed搜索；先完整图身份、官方反向重叠及数值/资源验收，再仅按完整validation至少+0.5pp选择，冻结权重后test一次，官方有向评价与反向不重叠子集分开。当前只是审查记录，不是已就绪实现，活动时禁止新代码/配置热修改。

WN继续审查特征语义、图/PPR方向与预训练覆盖；旧Books/Cloth固定seed池是额外容量适配，seed池大小不等于训练所见全部上下文节点，缩小数据不能推导更接近。不要再次运行已完成bounded修复。前五已达标结果及所有旧实验保留。

以下为历史快照。
# 阶段切换：2026-10-14T09:49:55.6704024+08:00，Asia/Shanghai

Goodreads-LP全部636502节点1024维FP32编码已完成，inference_state.complete=true，日志记录本次恢复编码1058.4秒，CUDA完整冻结行缓存2.428GiB。3个校准查询的原路径/缓存路径行值、官方MRR/Hits及反向子集完全相同，仅是路径验收，不是论文测试成绩；证据当前run/cuda_row_cache_equivalence.json。现在同一session80479及run20261014T013045Z_lp_single_bo_cff256_s0_ad2b38继续预测头训练和完整官方validation，尚无首轮指标或最终test，不启动第二个GPU任务。

5项达标/3项未达标/2项未测计数不变，Books论文MRR×10010.42等待完整test；Products仍待校准。Products后续资源预案（未实现、未验证）：用两个完整1024维FP32磁盘数组做严格逐层全邻居推理，text第一层存A、text第二层存B、vision第一层覆盖已不再使用的A，vision第二层每batch与B的对应text终层融合后原位写B；vision第二层只读取A，因此不会读已融合的B作为邻居特征。临时峰值两矩阵约18.69GiB，加既有缓存是否满足30GiB上限必须实际预算核验。须验证作者GAT/LayerNorm/GELU/gate/MoE逐层与原完整两跳推理数值等价，逐阶段游标/RNG/身份、融合覆写可恢复性和磁盘完整性；不削减宽度/节点/官方测试，不减少邻居，不在Books活动时修改生产源码。该预案解决重复两跳计算/显存压力的可能性，不代表已通过容量或已具备结果。

以下为历史快照。
# 最新进度：2026-10-14T09:32:36.1016965+08:00，Asia/Shanghai

六个预声明修复候选全部结束，repair ledger=completed_bounded_validation_repairs、原session21618退出。全部候选源码/profile/作者文件、保存产物哈希、CSV/JSONL/TensorBoard/最佳epoch与图像核验；证据artifacts/audit/bounded_validation_repair_completion_verified_20261014.json。Sports验证选头后完整test已核验28.983557（-8.696443pp），WN两个候选不足0.5pp门槛复用77.377154（-4.592846pp）；Cloth两个候选验证0.235665/0.254040均弱于参考0.254387，复用原test22.442081（-6.437919pp），没有新test。仍5项达标、3项未达标、2项尚未测，不宣称修复成功。

GPU释放后审计旧LP原有源码全部未改，仅4个未导入独立修复模块新增；旧科学profile、KG参考SHA、原LP完成产物、冻结encoder及Books raw/prepared/cursor32前缀身份核验通过。父ledger SHA a9b5f9cf7b064fd1a357e3f9b784f86370c452359b2f30182d404a5bba4f8aa3，旧ledger和维护pause无损归档，迁移记录artifacts/audit/lp_resume_after_bounded_repair_migration_20261014.json。自建维护暂停已归档清除，不能误认为用户暂停或重新恢复repair流程。

scripts/run_remaining_lp_validation.py已恢复（session80479），新run20261014T013045Z_lp_single_bo_cff256_s0_ad2b38日志确认Exact inference resume cursor=32/636502；当前实际进程2160、RTX3060显存1109MiB，缓存实际cursor=65024/636502正常增长。只做Books seed0、全官方train-only图/full val/test/150负候选/1024维、validation选头test一次；Sports/Cloth基线复用不重训，前三个未达标需继续严谨审查。源/profile冻结，运行期间不热修改/并行GPU/删缓存。Products还待显存/磁盘/全邻居数值校准。

修复后续自审方向：对照附录C/E和原模型，核查模态语义/方向与PPR构图、全邻居推理等价、联合预训练数据覆盖/混合预算；这些是待检验假设，不能称为根因。任何新方案先写证据/未公开参数/资源/有限验证候选与验收，通过后冻结权重再做一次完整test；不能缩减官方测试集或依据旧test不断搜索。已达标四NC430及FB结果和原作者/全部历史保留，正式多seed重复由用户训练。监测提示已更新为Books活动阶段。

以下为历史快照。
# 最新进度：2026-10-14T09:13:48.6998896+08:00，Asia/Shanghai

用户再次要求继续修复，少量训练只能作为有依据的工程适配，官方验证/测试完整保持；没有承诺成绩。十个主任务目前8项已有完整测试：5项距论文2pp以内、3项未达标、2项未完成测试；若按达标目标计，尚有5项需要完成或修复。未完成测试是Goodreads-LP及Products，正式后六项多seed重复仍待用户训练。

Sports有限验证修复已选two-hidden-LayerNorm头：验证MRR0.320973（参考0.308242，超过预声明0.5pp门槛），保存权重锁定后完整test一次MRR×100=28.983557，较基线+0.680368pp，距论文37.68差-8.696443pp，仍未达标。37413全官方source/target/rank、150候选形状、6621反向重叠、重算MRR/Hits及非重叠子集、18轮验证日志/CSV/JSONL/TB、第8轮选择、最佳权重/父SHA均独立核验，PNG已目视核验。证据artifacts/audit/sports_lp_validation_repair_verified_20261014.json；不重复GPU评分。

WN两固定LBFGS候选验证79.334212%/79.466051%，较参考79.136454%提升都不足0.5pp，按预声明门槛拒绝，复用已核验测试77.377154%，不重测、不宣称修好。当前Cloth第一个修复候选运行，实际进程35252及RTX3060核实；不热改活动源码/profile、不并发启动。旧Books维护暂停/cursor32保留。有限头/优化器尝试若仍不足，下一步需审查特征语义/图与PPR/联合预训练覆盖的论文一致性，只用训练和验证证据设计新协议；不能靠减少测试数据或持续看test调参达标。前四NC430、FB及全部历史/作者文件保持。

以下为历史快照。
# 最新进度：2026-10-14 09:00，Asia/Shanghai

按已核验测试，目前十个主任务：5项距论文2pp以内（前四NC及FB），3项未达标（WN77.377154%，-4.592846pp；Sports MRR×10028.303189，-9.376811pp；Cloth22.442081，-6.437919pp），2项尚未完成（Goodreads-LP、Products）。Cloth97456全test/150候选/17360反向重叠/100轮CSV-JSONL-TB/选择epoch99/父SHA均独立核验、图像已目视核验；证据artifacts/audit/cloth_lp_single_seed_verified_20261014.json。作者文件、四NC430产物、FB与历史结果保留。

用户新要求优先实际修复未达标项，已启动scripts/run_validation_repair.py（session21618）；账本artifacts/workflow_state/validation_repair_current.json，预声明configs/validation_repair_20261014.json。WN仅换固定步长LBFGS数值求解、train-pair可逆affine（L2=0/1e-4两候选），保留冻结拼接linear、lr=.05/2000/100；正则化/optimizer为明确工程偏差。LP只试两个上游依据头/优化候选：existing2layer或two-hidden-LayerNorm、Adam5e-4/batch512/StepLR5gamma.1/30epochs/10patience；完整train-only图/官方full val/test/150候选/1024宽不改。依据固定官方MMGraph commit0769fa670c69a6dc0756a2d7fc697b980e494775，不冒称CAME作者LP设置。109全回归及实际新求解器CUDA/两个LP原路径-缓存路径等价通过，证据artifacts/audit/validation_repair_cuda_acceptance_20261014.json。只按validation至少提高0.5pp锁定，确认保存权重后test一次；不足则复用原结果，不为了test接近不断搜参数。

LP基线已经在Books首次编码cursor32/636502安全暂停（前一项Cloth已完整结束），pause文件remaining_lp_validation_pause.json为本次修复优先级维护请求，保留，修复期间不得自动恢复旧LP。历史Books缓存不删，之后续跑。新增修复模块未改任何旧生产源码；旧LP将来恢复需要GPU空闲后审计源集合新增并保留父ledger SHA迁移身份，不能改活动identity。当前先WN两候选，再Sports/Cloth；修复后继续Books/Products，完整重复仍交用户。

每阶段自审要核对目的/证据/不确定性/资源/验收；缺作者参数明示为重建，不把改善假设当已定位原因。WN第一fixed-step候选validation79.3342%，提升不足0.5pp，保持参考；第二候选运行中，尚无新的test。普通推进安静，阶段/确认结果/故障再通知。

以下保留历史快照。
# 最新进度：2026-10-14 08:21，Asia/Shanghai

Sports LP单seed完成并独立核验：official MRR×100=28.303189，论文37.68，差-9.376811pp（相对-24.8854%，未达2pp）。全部37413个官方测试source/target/rank，150固定负候选，6621反向重叠mask精确匹配；MRR/Hits及排除反向重叠子集MRR×100=27.658777独立重算一致。训练11轮、validation选择第1轮，test一次；训练loss下降而validation MRR下降，不应直接补十seed重复当作质量修复。预测头/负采样为明确未公开重建细节，后续协议审查只依据训练/验证证据，不依据test持续搜索。CSV/JSONL/TensorBoard、最佳轮、父SHA及产物哈希一致，曲线目视核验。证据artifacts/audit/sports_lp_single_seed_verified_20261014.json。

当前LP pipeline正常CUDA运行Cloth，账本remaining_lp_validation_current.json为running，run 20261014T000406Z_lp_single_cl_7a4775_s0_c2c99c；监测时约39轮、val MRR0.24356仍在提高，尚无测试成绩。Goodreads-LP将按冻结队列继续。活动源码/profile未修改、没有第二个GPU任务、前四项和两KG结果保留。KG完成分数FB93.481872%（-1.108128pp）、WN77.377154%（-4.592846pp）；Products仍未启动，六任务尚未全部达标。

以下为历史审计细节。
# 最新进度：2026-10-14 07:59，Asia/Shanghai

KG两项单seed已完整结束并独立核验：WN18RR77.377154%，论文81.97%，差-4.592846pp（未达2pp）；FB15K23793.481872%，论文94.59%，差-1.108128pp（达到本次单seed2pp目标）。分别3134/20466条官方测试pairs、labels、prediction逐项一致；验证选择锁、最佳权重/父SHA、JSONL/CSV/TensorBoard一致，训练曲线目视核验。FB两个affine验证候选90.384944%/93.264899%不如raw逆向94.006273%，按预声明验证门槛保持raw逆向最佳线性头，不由test分数选方案。账本kg_affine_screen_current.json=completed_two_kg_diagnostics；证据artifacts/audit/two_kg_final_confirmations_verified_20261014.json。仅一个下游seed，不能代替十seed正式统计。

下一阶段三个LP已启动：scripts/run_remaining_lp_validation.py，直接账本artifacts/workflow_state/remaining_lp_validation_current.json；顺序Sports/Cloth/Goodreads-LP，各seed0，冻结主encoder，不重训四NC。实际Torch+DGL CUDA、1024宽/150固定候选的行值和MRR/Hits/反向子集等价校验通过，完整104测试通过。配置configs/remaining_lp_single_seed.json已锁定ready/acceptance状态（活动时不得再修改此profile或任何现有源码）；新模块remaining_lp_pilot明确LP预测头/训练是重建设置，train-only官方图、full val/test、一次test和反向重叠单列保持。完整FP32矩阵缓存在GPU以减少重复传输，另留1.5GiB评分空间，不减维、不转CPU计算。证据artifacts/audit/lp_cuda_row_cache_acceptance_20261014.json及remaining_lp_official_data_checked_20261014.json。

用户无需额外点击启动第二份试验。3LP结束后Products仍待全邻居显存/磁盘校准，WN未达标如实保留；不得宣称后六项全部接近或整篇复现成功。前四项430结果及全部历史run/作者文件保留。监测新LP ledger、实际进程与run/缓存游标，不再恢复已完成KG流程。

以下为历史审计细节。
# 最新进度：2026-10-14 07:43，Asia/Shanghai

四项430结果已完成并保留，当前只做后六项各一完整seed诊断，正式重复训练留给用户。WN18RR最新验证79.13645%，冻结选择后测试77.377154%（距81.97%低4.592846pp，未达2pp），3134条测试对/标签/预测和选择锁独立核验完毕。FB15K237 raw逆向验证94.00627%，当前两个affine候选未全完成，最终test尚无，不得将val当test。

07:39本聊天诊断在137轮安全暂停，实际CUDA证明标准化原地操作与旧实现逐位相同、源缓存不变，校准临时显存峰值96MiB->32MiB；保留所有旧run并经有父SHA源码迁移后在新run 20261013T234217Z_kg_affine_FB_0c49e6_s0_c7cc07从138轮继续。前137条历史完全一致、父checkpointSHA一致，相关8项复测通过。观察每25轮约37.36秒->10.46秒，约3.57倍，显存观察5865->3794MiB；仅为短区间实测，不保证全程倍数。证据artifacts/audit/kg_affine_actual_speed_verification_20261014.json。当前工具session=89127，活动pipeline=scripts/run_remaining_kg_validation.py；不要再点第二份或热修改身份中的任何现有源码（现已包含remaining_lp_pilot.py）。

LP准备：新模块came_repro.remaining_lp_pilot和configs/remaining_lp_single_seed.json尚未实际CUDA运行；两个CPU适配测试通过、三个官方train图及全部val/test候选已独立核验。GPU解锁后先通过子任务内实际CUDA数值/排名等价检查，再串行Sports、Cloth、Goodreads-LP各seed0。完整每查询150负候选；GoodreadsLP test1710677条，约2.58亿正负对评分，比NC耗时大。LP头是显式重建选择，Products尚待完整全邻居推理容量校准。后六项未完成、不能保证全部距论文2pp。

下面保留本次及此前审计细节。
# 当前复现状态（2026-10-14，Asia/Shanghai）

## 后六项单seed诊断与断点核验：07:23

用户最新授权：后六个主任务各做一个完整seed的真实评估，尽量距论文2个百分点以内；完整重复训练由用户自己实施。2pp是目标，未达标不得伪称已接近。前四项430/430已全部完成并保护，Ele-fashion84.1524%（-0.9776pp）、Wiki-CS400次均值76.6633%（-1.9067pp，跨split/seed合并SD）、Arxiv70.9639%（-0.6961pp）、Goodreads-NC82.0995%（-1.1105pp）。最终产物在artifacts/final_results/four_node_tasks，原作者文件/预训练模型/全部四项结果不改。

WN18RR原有向试跑76.3880%；按验证集选择训练边逆向视图后77.1219%，论文81.97%，差-4.8481pp，仍未达标。FB15K237原图验证90.6074%，逆向候选日志至1000轮验证92.4380%，还没有最终测试。2026-10-14检查没有任何Python训练进程；旧KG账本running是陈旧状态，退出原因尚不能确定。完整保存断点仅975轮、next_epoch=976，optimizer/RNG/游标/975条历史完整，SHA核验；不能将1000轮日志等同1000轮完整持久化。已恢复到新run 20261013T232237Z_kg_graph_FB1_3e70ca_s0_0ec642，旧run保留。证据：artifacts/audit/kg_interruption_resume_20261014.json。

当前入口scripts/run_remaining_kg_validation.py，本聊天已启动，用户暂不要再点第二份。它只处理两个KG的seed=0：先完成图候选验证，再比较预声明的训练对统计affine数值条件候选，每项原图/训练边逆向两个候选、lr=.05、max2000、patience100；按验证分数至少提升0.5pp选择适配，否则保留raw参考。affine是明确工程适配，最终折回原始头尾拼接上的线性分类器，不能声称作者规定了标准化。冻结1024维编码器不重训，图只用训练边，完整官方划分不缩减，测试分数不参与选择。新ledger为artifacts/workflow_state/kg_affine_screen_current.json，所有源码/编码器身份冻结；不得热修改其现有源码。

测试确认直接加载选定验证run的最佳分类器并重放验证分数，锁定分类器/摘要SHA后测试一次，保存全测试预测并独立核验；不再为了确认重新训练同一个2000轮头。已有WN原协议测试经核验复用，不重复测试该配置。真实Torch/DGL CUDA、完整100项回归通过。四个LP/Products尚未启动，KG诊断结束后须继续审查LP吞吐、官方候选/反向重叠及Products内存/缓存预算，不能宣称后六项完成。

## 已完成清理与保护

四项430个run、430个checkpoint目录及3819项产物SHA建立保护清单，全部官方测试预测独立重算。只清理两个未被当前结果引用的旧派生编码缓存，释放约2.66GiB；19个旧PyCharm运行配置无损归档到docs/history，作者代码、原始数据、训练模型/日志/图像/预测不删。清理后3819项哈希无变化，证据artifacts/audit/cleanup_integrity_verification_20261013.json。之后不再为视觉整洁删除实验历史。

以下为历史快照，旧计数不代表实时进度。
# 当前复现状态（2026-10-13，Asia/Shanghai）

## 主入口实际运行验收：2026-10-13 07:32

用户已通过指定基础Python39启动 `scripts/run_00_main_pipeline.py`，实际转交项目环境并串行使用CUDA。Wiki-CS split=0/seed=1已完成：测试77.6124508%，与已确认seed=0分数相同。5847个完整官方测试节点/标签逐项核验，JSONL/CSV/TensorBoard的181轮step和loss一致、数值有限，PNG已目视检查，父检查点SHA核验通过。总控制器自动继续seed=2，当前完成14/430个结果。验收证据为 `artifacts/audit/main_entry_live_acceptance/verified_resume_and_completion.json`；不能由此承诺以后每个seed或每种硬件状态永不出错。

验收观察器额外启动被同入口的并发锁拦下，未启动第二份GPU训练；观察器误将暂停请求施加到已由用户启动的任务，该任务在epoch=1安全保存，随后用户运行入口恢复到epoch=2并完成。失败的额外启动及观察器记录保留，不冒称其自身验收成功。当前只读核查用户运行，生产源码/配置没有热修改，指纹与活动账本一致。

## 最新单入口操作

按用户最新要求，前四项只需在PyCharm直接运行 `scripts/run_00_main_pipeline.py`，不填参数；自动复用第一个已完成的十seed和三个已确认seed=0结果，然后补齐当前固定协议的全部四项重复实验，完成后结束。其他六主任务/few-shot/消融留待用户训练完再处理。`run_full_suite.py`是兼容别名，无需两个都点；稳定性小入口仅作为可选分段诊断保留，不是必要前置步骤。暂停后仍点击同一个主入口恢复。配置、生产源码及已生成账本身份本次未改变，仅统一操作文档和PyCharm入口名称；本轮没有启动正式训练。

## 四项固定协议交接：2026-10-13

用户当前优先完成Ele-fashion/Wiki-CS/Arxiv/Goodreads-NC四项，完整训练由用户自己点击运行。新入口和预备账本已实现并验证；本轮没有启动正式新增seed训练。Ele-fashion十次结果和另外三项已确认seed=0结果，共13个真实官方测试结果，逐项核验预测/标签/节点ID/分数及产物哈希后直接复用。旧有向图下的低分重复实验不混入修正后的统计，作者文件及历史实验保持原样。

先运行 `scripts/run_four_node_stability.py`：三个修正任务各补seed=1、2，Wiki官方split=0，共6个新作业。随后运行原 `scripts/run_full_suite.py` 或 `scripts/run_00_main_pipeline.py`：两者默认现已转向修正的四项重复流程，包含Ele-fashion10次复用、Wiki20×20、Arxiv10、Goodreads-NC10，共430个结果，初始待417个；完成稳定性检查的6个结果将核验后接入，不重复训练/测试。新的主账本指针分别为 `corrected_node_stability_current.json` 和 `corrected_node_suite_current.json`。暂停入口和操作见 `docs/corrected_four_node_tasks.md`。

编码器SHA、双向去重策略、Adam/lr=.05/2000轮上限/100patience、官方划分和CUDA路径已固定，无额外预训练、无24小时硬截止；同数据集重复seed共用完整编码，派生缓存上限3GiB。作者原代码、论文和原始数据不改。另修正续跑后的推理进度打印判据，避免游标未对齐新批量时日志长期静默，推理数学及缓存身份不变。

验收：全量94项测试通过（含实际CUDA/恢复已有检查），新的6项检查覆盖完整430网格、预声明稳定性子集、图/seed/epoch非法改变拒绝、缺官方测试节点/虚假accuracy拒绝、同编码器头检查点续跑、13个历史完整官方测试结果核验。两个prepare-only成功，用户指定基础Python39运行原主入口dry-run成功转交项目环境，确认417待完成、零新增预训练、Ele-fashion不重训。比较图已目视核验，与CSV源数值一致。重复seed/不同Wiki split的实际波动尚未测，不能保证所有分数或最终均值与单seed完全相同。

## 单seed图处理确认完成：2026-10-12 22:23

Goodreads-NC 全部685294节点编码已完成；seed=0分类器按验证准确率选定epoch=1999，在2000轮上限内结束。正式独立测试确认82.096368%，论文83.21%，数值差-1.113632pp。保存预测逐项对齐全部203428个官方测试节点及标签，独立重算准确率与summary完全一致，test_evaluations=1。核验见 `artifacts/audit/goodreads_fast_confirmation_verification.json`；确认run为 `runs/20261012T141502Z_quality_came_c83ea6_s0_ede6f3`。

当前快速账本为 `completed_single_seed_graph_screen`，没有启动剩余六任务入口。Ele-fashion已有十seed结果不变，Wiki-CS、Arxiv、Goodreads-NC三项修正后单seed测试均距论文均值约0.74至1.11pp；这些数值不能代替多seed/多split统计。论文主任务仍剩Products节点分类、Sports/Cloth/Goodreads链接预测、FB15K237/WN18RR关系分类六项单seed测量，重复实验、few-shot及消融另计。已目视检查更新后的comparison.png，其验证与测试分开显示，数值与保存结果一致。

## 最新修正与测试确认：2026-10-12 21:36

Ele-fashion 保留已有十seed结果84.1524%，论文85.13%，未重训。Wiki-CS已完成修正后的seed=0/split=0测试77.6125%，论文整体78.57%，数值差-0.9575pp；Arxiv已完成修正后的seed=0测试70.9154%，论文71.66%，差-0.7446pp。两项均先按验证集选定双向/去重图处理，再做一次测试确认；仍是单seed诊断，不代替论文多seed/多split统计。

Goodreads-NC仍在完整图编码，尚无修正后的测试分数。实际CUDA压力校准通过：128与8的输出最大差约8.2e-7，压力样本峰值2387.49MiB。推理批量从8改128；缓存已编码前缀52696行保留，经审计迁移后继续，同一原模型/原始数据/官方划分不变。所有旧账本与427个已完成旧作业保留，原队列继续暂停。

当前主入口：`scripts/run_fast_graph_screen.py`。它复用八个子集轮次的原模型，追加预训练更新数为0；没有累计24小时截止，Ele-fashion不改。结果位置以`artifacts/workflow_state/fast_graph_screen_current.json`为准，目录中`results.json`/`.csv`和`comparison.png`分别保存数值及验证/测试分开的对照图。旧`run_quality_pilot.py`的额外一轮方案保留为可选历史设计，不是当前优先入口。

NC修正完成并审查后，`scripts/run_remaining_single_seed.py`测六个剩余主任务，各seed=0，不自动启动十seed/消融。所有尚未测完的任务不能保证接近论文，只有保存预测并核验的实测结果才计入达到目标。完整正式复现范围仍含重复实验、few-shot和消融。

实际新增工程验收：完整回归86项通过；新快速选择及真实预测核验2项通过；真实CUDA续训3次更新验证Adam/RNG/采样游标继承及日志一致，属于功能验收、未用于替代主模型。作者文件哈希未改变。批量校准与迁移证据见`artifacts/audit/books_inference_batch_calibration_128.json`、`fast_graph_batch_resume_migration.json`。

## 最新质量审查快照：2026-10-12 18:48

当前账本完成 425 个作业，正在 Goodreads-NC seed=4。常规子集联合预训练已完成；Ele-fashion 10/10 测试均值 84.1524%，Wiki-CS 400/400 均值 70.8898%，Arxiv 10/10 均值 58.1195%，Goodreads-NC 4/10 暂均值 68.3786%。论文对应 85.13%、78.57%、71.66%、83.21%。只有 Ele-fashion 接近，不能宣称整体论文数值已复现。

已审查完整预训练损失与梯度日志：三域损失下降；未发现文本嵌入全零，不足以证明迁移表征达标。新的质量恢复顺序见 `docs/accuracy_recovery_plan_20261012.md`：先本地 CLIP 对照与推理/划分一致性，再核对图协议，最后才试有限预训练预算候选。DGL Wiki-CS 官方加载实现使用 `to_bidirected`，与当前保存的原始有向视图存在需要核查的差异；未证明这是分数差距原因，也未改图。

本轮仅新增方案并更新状态文档。作者文件 SHA256 核验一致，生产源码/配置/模型/历史账本未改，CUDA 诊断对照与准确率修正尚未执行。当前运行保留。下面的内容为此前各次快照和工程验收，旧计数不代表实时进度。

## 此前进度与工程验收

最新实际进度及 I/O 修复：第一套常规联合预训练于 20:04 北京时间完成 10240 次 CUDA 更新、8 个固定子集轮次；模型已保存。随后 Ele-fashion 全邻居推理因 Windows WinError 5 拒绝原子替换 inference_state.json 而退出，已提交 cursor=24960，模型没有丢失。不能认定具体是哪一个进程占用文件。

已统一状态 JSON、任务 ledger、暂停请求和模型 checkpoint 的原子保存：同目录唯一临时文件、flush/fsync、Windows 5/32/33 错误最多 5 秒退避重试；持续故障保留旧有效目标并报错，不删除目标或跳过节点。JSON 读取也有同类短暂冲突重试。79 项完整测试通过，包括真实 Windows 读取句柄阻止替换、释放后成功恢复、持续占用保留旧 cursor、磁盘满立即失败及 CUDA 状态恢复。

已按原配置从 cursor=24960 完成 Ele-fashion 全部 97766 节点的实际 CUDA 推理，以及 seed=0 冻结 probe（max_epochs=2000、patience=100、lr=.05）：验证选择 epoch=506，early stop 于 epoch=606，测试 accuracy=84.1391067167%，只完成单个下游 seed，不是多 seed 最终均值。生产预训练未重跑；旧 ledger/模型/失败 run 保留，I/O-only 审计迁移保留累计时间与已完成任务，新验证耗时也计入预算。当前已完成作业 2/1471；下一项为 Ele-fashion seed=1，继续运行 scripts/run_full_suite.py 即可。真实验证 run：runs/20261011T121840Z_came_node_el_60a960_s0_0d81f0；验收：artifacts/audit/atomic_io_repair_acceptance.json。

注意旧摘要 pretraining_complete=false 指旧全节点 epoch 判据，不能用它判断声明的子集预算是否完成；对应主模型 summary 的 completed_subset_rounds=8、complete_declared_subset_pretraining=true。完整论文数值复现仍未完成。

最新运行修复：用户已启动全任务生产入口，十域数据与实际 Torch/DGL CUDA 预检通过；第一项训练在 TensorBoard 初始化失败，事件文件路径为 261 字符，尚未进行生产 optimizer 更新。已采用 Windows 原生扩展路径，并压缩新 run ID，完整语义名称仍在 config.json；同时修复子作业失败时总入口错误返回 0 的问题。旧目录不重命名。完整 logger 写读、同一 suite_job 路径下三步三域 CUDA、TensorBoard loss 与 JSONL 一致均通过；最新完整测试 75 项通过。

旧失败 ledger 与累计时间已原样保留，严格核验无训练检查点、科学配置/数据/环境未改变后建立有父哈希的新 ledger，并更新当前指针；未重置一天预算，未跳过第一项任务。验收见 `artifacts/audit/tensorboard_logging_repair_acceptance.json` 和 `tensorboard_ledger_migration.json`。用户继续运行 `scripts/run_full_suite.py`；没有代替用户重启完整生产训练。

最新用户需求为全任务/重复实验、一日预算、安全暂停续跑。已新增 **CAME_Full_Suite_Resume**（`scripts/run_full_suite.py`）与 **CAME_Request_Safe_Pause**：11 个固定子集预训练作业、1,460 个评测作业，涵盖十域主任务、三域 few-shot、CLIP 基线、九个四域消融；其他域十个下游 seeds，Wiki-CS NC 20 split×20 初始化。默认仅一个预训练 seed=42；不冒称十次独立预训练，也不冒称全部竞争方法与附录分析都已实现。累计 24 小时只约束预算，不证明完整网格可完成。操作和剩余覆盖缺口见 `docs/full_suite_resume.md`。

安全暂停代码已添加到预训练 optimizer 边界、NC/KG/LP 完整轮次边界、PPR 缓存提交和推理游标；恢复创建新 run，并自动传入相应 checkpoint，已完成结果经哈希核验后跳过。每个 child 自持 GPU 锁，防止父进程被强制停止后并发。整个项目最新完整测试 70 项通过，含真实 CUDA 分类/LP 头在首轮及最后一轮暂停后恢复的连续性测试；后续汇总与预训练暂停审计见 `artifacts/audit/full_suite_resume_acceptance.json`。

最终补充验收完成：完整测试 72 项通过；总入口状态记录调整后相关 8 项复测通过。真实三域 CUDA 从旧 step=12 检查点训练至 step=14 安全暂停，再从该检查点恢复，仅记录 step=15/16；optimizer/RNG/固定池/cursor 保存，parent SHA256 对齐，日志 JSONL/CSV 一致，TensorBoard 与暂停/恢复 PNG 已核验。对应 run 为 `20261011T102253Z_subset_offline_cuda_calibration_seed42_0eb8d7` 和 `20261011T102454Z_subset_offline_cuda_calibration_seed42_a1c586`；验收报告已落盘。默认 Python39 的完整网格入口 dry-run 成功。全部生产任务尚未启动，未证明一天完成或论文效果。

当前范围：用户此前全节点重建训练已在 step=546 中断，历史日志/检查点保留；没有完成正式 8 轮，也没有复现论文数值。已重读全部 19 页，确认第 16 页附录 E 的 Products 子集预训练与离线 PPR，纠正之前过重的全节点预算。修订前状态保存在 docs/history/current_status_before_subset_revision_20261011.md。

## 一天策略修订

新增 `subset_offline_train.py`、`subset_day_pipeline.py`、`scripts/run_subset_day_pipeline.py` 与 PyCharm **CAME_Subset_Offline_24h**。保留作者完整模型、三域联合预训练、8 个固定子集轮次；Products/Books-LP/Cloth 固定池为 16384/16384/8192，batch=32，共计划 10240 个 batch。具体池大小与 Books/Cloth 限额是公开声明的重建/容量适配，不是作者给定值。

所有选定 seed 的 PPR 邻居 ID 在优化前离线准备，训练期间缺失缓存明确拒绝；每步仍组装 induced union topology。子集轮次单独记录为 subset_round/completed_subset_rounds，completed_epochs=0，禁止与全节点轮次混淆。日志、指标、TensorBoard、完整 RNG/采样游标检查点和诊断图继承原设施。

主模型含离线准备最多 8 小时；优先 Ele-fashion 的 CAME、CLIP-text、CLIP-image 各 3 个下游 seed，保留最多 2000 probe epochs/lr=.05/validation 早停。两个同池同轮次消融各最多 4 小时，仅完整完成才作可比评测。24 小时是总监控预算，不能承诺所有任务完成或只差几个百分点。细节、证据、未披露项和省略范围见 `docs/paper_protocol_reread_20261011.md`。完整生产训练尚未启动。

实际新协议 CUDA 验收：`runs/20261011T095314Z_subset_offline_cuda_smoke_seed42_19b52e`，三域各 16 个 seed、三次真实反向更新、53,617,687 参数；峰值 allocated 1609.6 MiB。此小池验收不是生产预算、长程质量或全部批次显存的保证。正式 batch=32 的补充短程与最新测试记录见 `artifacts/audit/subset_day_acceptance.json`（生成后为准）。

最新验收已完成：`runs/20261011T095723Z_subset_offline_cuda_calibration_seed42_d14bb8`，batch=32、每域 128 seeds、共 12 个真实 CUDA 更新，最大 allocated 2725.8 MiB；JSONL/CSV 原值一致、TensorBoard 存在、loss/专家图已目视检查。完整测试 60 项通过，最终受影响测试 9 项通过。另以每域 64 个独立无标签节点测未缓存 PPR，按均值外推固定池约 22 分钟计算时间，不能视为全流程或最慢节点保证。默认 Python39 启动新入口的 dry-run 成功，完整一天生产训练未启动。验收报告已写入上述文件。

## 环境与作者文件

- 用户默认 Python 是 C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe（3.9.13）；实验使用 .venv-came39\Scripts\python.exe。
- RTX 3060 Laptop 6 GB；PyTorch 2.1.2+cu118、DGL 1.1.2+cu118。最新预检实际执行两者 CUDA 运算及完整作者模型反向/优化器更新。
- model.py、fusion_modules.py、论文初始 SHA256 均一致。完整主模型 53,617,687 参数，hidden=1024、每视图 6 个专家（总 18）、top-2，损失权重 .5/.1/.5。
- 未使用的缺失 unigraph2.MoE 导入只在内存移除。作者文件未改写；Windows 与论文服务器的软件栈差异持续记录。

## 十个完整数据域已准备

五个多模态域：Ele-fashion、Amazon-Sports、Amazon-Cloth、Goodreads-LP、Goodreads-NC。19 个必要资产合计 9,926,641,608 bytes，固定 MM-Graph revision c707a3203b77a153f11370a05d2a9be6571f7670，SHA256 与发布者 LFS 一致。

另外五域：ogbn-Products（2,449,029 节点）、ogbn-Arxiv（169,343）、Wiki-CS（11,701）、FB15K237（14,541）、WN18RR（40,943）。完整图、官方划分、文本及 FP32/L2 CLIP 文本特征已落盘；所有 encoding cursor 到达完整节点数，特征 SHA256 已复核。没有以子集或 OGB 数值特征代替 CLIP 文本特征。

OGB 保留官方 node mapping。Products 原边数 61,859,140，按官方 inverse-edge 规则的有向存储数为 123,718,280，不能混淆这两种计数。OFA 来源固定 9045ccdfa2d02fd55841700e7d72ef563bbc2de9；Wiki-CS 固定 f5207315d649377f936edb66d7d93f5342f01d81。KG 图只用训练三元组，实体文本不混入关系标签。

HF 文件使用发布者校验值。OGB/Google Drive/源码 ZIP 没有可用发布者 SHA256 时，记录官方来源、固定版本和本地 SHA256，不把本地指纹伪称为发布者校验。

## CLIP 特征语义

- CLIP 固定 openai/clip-vit-base-patch32 revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268，模型/tokenizer/config 指纹保留。
- 五域各 64 个随机文本节点独立重编码，确认前 512 维为归一化文本，最大误差小于 7e-7。Goodreads-LP 的发布 node_mapping 是旧版 2,360,655 节点映射；最终 raw text 行序含 636,502 节点，通过随机节点特征匹配确认对齐，未盲用旧映射。
- 五域各 8 张完整图片样本匹配后 512 维。三个商品域使用默认 CLIP 图像预处理；书籍两域需 PIL BICUBIC 直接缩放到 224×224 后归一化，默认中心裁剪不能匹配。
- 图片核验仅下载固定版本归档首 16 MiB，保留 prefix/单图 SHA256。没有下载完整图片归档，也不把 prefix 指纹冒充完整归档校验。训练使用已有发布者校验值的完整特征。
- 文本-only 图复制文本作为视觉占位来自论文 §5.2。新域的文本拼接/截断细节仍标为重建选择，不声称掌握未公开作者生成脚本。

## 训练与评测设施

常规语料：Products + Goodreads-LP + Amazon-Cloth。Few-shot 语料：Products + Goodreads-NC + Amazon-Cloth。旧 `joint_*_reconstruction.json` 为 8 个全节点轮次、max_steps=null 的过重重建，保留但不再视为论文必要默认预算。当前一天入口使用上文固定池、离线 PPR、8 个子集轮次。PPR top-128，真实 PPR；alpha/epsilon、混合方式、seed batch=32 等未披露项继续显式声明。

已有冻结 NC、KG head/tail 关系分类、LP 官方负样本 MRR/Hits 与反向非重叠子集、5-way 1/3/5-shot、CLIP-text/image 基线、九种真实实现的消融配置。Wiki-CS 使用官方 stopping mask 的 loss 选轮次，另报 validation 指标，保留全部 20 split；其他分类任务按 validation accuracy 选模。LP head、few-shot 匹配/查询策略、probe optimizer/patience 属于公开标注的重建选择。

联合训练 checkpoint 保存 RNG、每图 order/cursor、optimizer。分类与 LP head 保存 epoch-boundary 状态并提供 --resume-head；恢复新建 run，核验配置/数据且记录父检查点 SHA256。日志包括 human logger、JSONL/CSV/TensorBoard、三对 loss、梯度、零嵌入、显存/耗时/采样覆盖。PNG 标明来源和范围，并保留对应原始数值。

## 实际验收

- 常规六步：runs/20261010T035850Z_joint_representation_reconstruction_seed42_ec0e6e；恢复累计九步：runs/20261010T040056Z_joint_representation_reconstruction_seed42_7d8c82。三域各 96 个种子，completed_epochs=0；最高 torch allocated 2616.5 MiB。
- Few-shot 三步：runs/20261010T040341Z_joint_few_shot_reconstruction_seed42_17199c。三域各 32 个种子；最高 allocated 2702.9 MiB。含 PCA、专家使用率、数值诊断和训练图。以上都是有限预算验收。
- 完整 Wiki-CS CAME 冻结推理＋两轮 probe：runs/20261010T040947Z_came_node_wiki_cs_seed0_f62eaf，stopping mask 选模通过；编码器只训练九步。更早 val-mask 工程 run 保留，不能混入新协议正式比较。
- 原 Ele-fashion 40 步 pilot、恢复链和 84.064% 单 seed probe 原样保留，仅为工程证据。未裁剪异常大梯度 run/失败负控制保留审查注记；裁剪 1.0 始终记为工程偏差。
- 十域预检依据是 artifacts/audit/preflight_reproduction.json；最终测试、图片/数值验收及最新运行目录见 artifacts/audit/verification.json 和 artifacts/audit/preparation_acceptance.json。

最终验收：33 项测试通过，8 个实际有限预算 run 的 JSONL/CSV、logger/TensorBoard、checkpoint/游标/父哈希、预测/rank/episode 与 PNG 通过审查。新增实际分类头恢复至第三轮：runs/20261010T125902Z_came_node_wiki_cs_seed0_c0b493；官方 Sports LP 头恢复至第二轮：runs/20261010T130101Z_clip_text_lp_sports_copurchase_seed0_89202e。关系分类单轮验收：runs/20261010T130117Z_clip_text_relation_fb15k237_seed1_be968a；few-shot 日志/图：runs/20261010T125640Z_clip_text_few_shot_wiki_cs_seed0_ca5b1d。全部是前期链路验收，不是完整预训练成绩。

## 空间控制

原生 torch.load(..., mmap=True) 的位值一致性已验证，删除五份重复派生特征共 6,535,787,136 bytes（6.09 GiB）。官方 raw 和历史实验不删。未使用 .venv-came 3.11、.runtime/python、.runtime/uv-cache 已清理；用户原 .venv 和指定 .venv-came39 保留。

E 盘存数据/代码/日志/图；新大 checkpoint/可再生嵌入在 configs/storage.json 指定的 C 盘项目专属缓存。项目上限 40 GiB、外部缓存上限 30 GiB，磁盘余量至少 3 GiB。原子 checkpoint 临时副本也计入预留；run/checkpoints/location.json 指向实际文件。实时容量见 artifacts/audit/storage_budget.json，不能沿用旧读数。

## 未完成与容量边界

基础 Python 点击启动问题已修复：scripts/_run_environment.py 在所有 run_*.py 入口导入训练依赖之前，将基础 Python39 启动显式转交至已有项目 .venv-came39。已经实际使用用户报错中同一基础解释器及总脚本验证 --dry-run 成功，也验证从 scripts 工作目录启动；没有代替用户启动完整训练。CUDA/训练轮数/原作者文件不变，环境不存在或子进程失败时明确停止。

新增唯一主流程入口 scripts/run_00_main_pipeline.py / CAME_00_Run_Main_Pipeline：自动顺序调用 01→02→03→04→05，保留两个语料各 8 轮，不隐式启动基线/消融；失败即停。只验收了 dry-run 和相关控制测试，没有启动完整流程训练。

PyCharm 补充交接：新增 scripts/run_01..run_07 与 run_tensorboard.py；.run/ 中八个共享运行配置指定项目 .venv-came39、项目工作目录和 UTF-8。入口不改变原训练协议，主预训练仍为 8 epochs/max_steps=null，辅助 CPU/磁盘工作保留、模型运算要求实际 CUDA。最新完整测试 43 项通过，覆盖正确语料完整 encoder 选择、十 seed/二十 split 网格、OS GPU 入口互斥和已完成子任务产物验证。真实点击入口等价预检 runs/20261010T133324Z_pycharm_preflight_db5446 成功；未实际操作 PyCharm GUI 按钮。详细顺序见 docs/pycharm_quickstart.md，审计见 artifacts/audit/pycharm_launchers.json。

两套 8 轮预训练、多 seed 主结果、所有基线/消融的数值实验尚未执行，留给用户自行训练。作者只给两份代码和论文；未披露 PPR 种子预算/混合/优化器/LP head 等不能保证恢复作者原始设置。

6 GB 显存短程通过不保证所有批次和大图全邻居推理均可承受。每套全节点种子预算约 80 万个预训练 batch，可能持续多天。密集节点全邻居推理有容量限制，不会自动换 PPR 推理、降维或 CPU。操作与显式适配边界见 docs/training_handoff.md。

Goodreads-NC 官方 7,235,048 条边比 PDF 表格少 36；7,372 个未在官方标签划分的节点保持原样。LP 训练拓扑只含训练边，反向重叠风险继续依据 artifacts/audit/lp_split_overlap.json 分开报告。


## 2026-10-14 07:36追加实测（参照本文顶部阶段）

WN18RR两个affine候选完成，逆向affine的验证79.13645%，比raw逆向78.24654%高0.8899pp，达到预声明0.5pp门槛；原图affine不优于参考。冻结选择锁后直接测试保存的折回线性头：77.377154%，论文81.97%，差-4.592846pp，仍未达2pp目标。独立核验3134条官方测试pairs/labels/prediction完全匹配，test一次，折回validation预测差异0，证据artifacts/audit/wn18rr_affine_confirmation_verified_20261014.json。此结果不能宣称WN成功复现。

FB15K237 raw逆向验证完成2000轮，best epoch1986，val94.006273%，尚未最终test；当前正在完成预声明affine候选。

三个LP官方train图/val/test候选均CPU独立核验，图边与官方训练正边按顺序精确一致；每条查询固定150个负候选（不是假设1000）。验证/测试查询分别Sports45557/37413、Cloth121323/97456、GoodreadsLP570464/1710677。GoodreadsLP完整一次测试要评分约2.58亿正负节点对，所以不能以NC的小分类头耗时预测LP耗时。FP32缓存大小分别205824000/515436544/2607112192 bytes。证据artifacts/audit/remaining_lp_official_data_checked_20261014.json。新增LP模块尚未启动实际CUDA校准，不可说它已通过真实吞吐验收。

正在运行的KG身份未包含之后新加的LP模块；原有KG源码SHA全无变化，新增非依赖模块的审计见artifacts/audit/kg_inactive_source_additions_20261014.json。如果活动进程退出后需恢复，先审计其源文件集合新增再迁移身份；不得修改活动ledger identity或已有科学参数，追加审计需保留父ledger SHA。

















实时阶段补注 2026-10-14T17:53:22.8137917+08:00：K4实际Torch+DGL CUDA与seed1完整optimizer/RNG暂停恢复逐位一致均通过，证据hard_lp_CUDA_acceptance_20261014.json。新Sports validation-only run20261014T095231Z_hard_lp_spor_822192_s0_b30b56已实际启动PID22544，首轮validation0.3553845136，最多60轮；还没有该候选新test。Cloth下一串行，不启动用户正式重复。
