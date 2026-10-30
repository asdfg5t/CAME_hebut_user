# HEBUT CAME 论文复现项目

[English](README.md) | [简体中文](README_zh-CN.md)

本项目为论文 **A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs（CAME）** 的独立复现工程，由用户维护，非论文作者官方仓库。

## 范围与状态

作者提供的 `model.py`、`fusion_modules.py` 保持原始字节。新增实现位于 `src/came_repro/`，入口在 `scripts/`，配置在 `configs/`。

最新状态文档更新至 **2026-10-22**，以下不是上传时重新测量的结果：

- 八项主任务已有下游重复记录：Ele-fashion、Wiki-CS、Arxiv、Goodreads-NC、FB15K237、Amazon-Sports、Amazon-Cloth、WN18RR，共 470 个结果。
- 下游重复复用预训练编码器，不等于多次独立预训练；部分指标仍低于论文。
- Goodreads-LP 有独立协议重建工作，文档未证实其全部正式重复完成。
- Products 十个分类头已完成，完整测试在 427,136 / 2,213,091 行处暂停，最新状态尚无最终测试准确率。
- 基线、few-shot、消融已有入口，实现或 dry-run 通过不代表数值实验完成。

证据与限制见[当前状态](docs/current_status.md)。本次发布仅包含源码和文档，**不包含本机实验结果与检查点**，不宣称整篇论文全部成功复现。

## 环境安装

本机验证环境：Windows、Python 3.9.13、PyTorch 2.1.2+cu118、DGL 1.1.2+cu118、RTX 3060 Laptop 6 GiB。该环境与论文服务器不同。模型计算要求 **Torch 和 DGL 真实 CUDA 运算**，不会自动降级至 CPU。

原工作站执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
& .\.venv-came39\Scripts\python.exe scripts/check_environment.py
```

安装脚本固定使用 `C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe`。其他机器请指定自己的 Python 3.9 路径：

```powershell
& 'C:\path\to\Python39\python.exe' -m venv .venv-came39
& .\.venv-came39\Scripts\python.exe -m pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
& .\.venv-came39\Scripts\python.exe -m pip install -r requirements/phase_02_windows-cu118.lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
& .\.venv-came39\Scripts\python.exe -m pip install -e . --no-deps
```

## 数据与运行

1. 阅读[复现计划](docs/reproduction_plan.md)、[环境说明](docs/environment.md)、[当前状态](docs/current_status.md)。
2. 单独取得用户提供的论文，放在根目录并命名为 `CAME 4545 ICML 2026.pdf`。SHA256 须匹配 `artifacts/audit/original_files.json`；仓库不分发 PDF，但原文件保护检查仍要求它存在。
3. 根据计划使用 `scripts/download_datasets.py`、`scripts/fetch_reproduction_assets.py` 及各阶段准备入口。保留官方版本、发布者校验值和原始文件。克隆不会自动下载数据，数据准备需单独安排容量和时间。
4. 检查 `configs/storage.json` 等配置中的本机路径。历史账本与冻结交付清单是本地记录，不是可移植发行包；新机器须重新准备依赖，不能伪造完成账本绕过检查。
5. 按阶段选择入口。`run_00_main_pipeline.py` 默认只管前四项节点任务。Products 使用 `run_24_products_sampled_stream.py`；25–37 对应基线、few-shot 与消融。这些流程依赖准备好的本地输入，仅复制源码不能恢复历史实验。

前四项流程的计划检查命令，不启动训练：

```powershell
& .\.venv-came39\Scripts\python.exe scripts/run_00_main_pipeline.py --dry-run
```

导航和迁移限制见[仓库指南](docs/repository_guide.md)，不要只按编号启动历史训练入口。

## 实验规则

- 两层 GAT、1024 维；每视图六专家，对应作者参数 **总计 18 个**。损失权重 `w12=0.5`、`w13=0.1`、`w23=0.5`。
- 论文设置与工程适配单列，固定子集轮不冒充全图 epoch。
- LP 拓扑只用训练边；官方有向划分有反向节点对重叠，去重叠评测单独报告，见 `artifacts/audit/lp_split_overlap.json`。
- 验证集选模、测试集不调参；保留 Goodreads-NC 官方排除节点和比论文表格少 36 条的边数。
- 续跑创建新 run，记录父检查点哈希，保留优化器、RNG、采样游标，不覆盖历史实验。

## 检查

```powershell
& .\.venv-came39\Scripts\python.exe -m pytest tests/test_run_environment.py tests/test_paper_task_groups.py -q
```

以上为针对性 CPU 检查，不代表 CUDA 训练、数据准备或论文精度验收。其他测试可能依赖本地数据、历史清单或 GPU。

## 许可证与贡献

[LICENSE](LICENSE) 的 MIT 授权仅覆盖原创复现基础设施；作者原代码、论文等排除项见[第三方声明](THIRD_PARTY_NOTICES.md)。MM-Graph 保留上游 MIT 许可。

另见[贡献说明](CONTRIBUTING.md)与[清理记录](docs/cleanup_20261030.md)。
