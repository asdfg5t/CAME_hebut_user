# Repository guide / 仓库指南

Read the root [English README](../README.md) or [中文 README](../README_zh-CN.md), then the newest entries at the top of `current_status.md`. Older entries may be superseded.

先读根目录 README，再看状态文档顶部最新记录；旧说明可能已被替代。

## Directory map / 目录用途

- `model.py`, `fusion_modules.py`: preserved author code / 原作者代码。
- `src/came_repro/`: data, sampling, training, evaluation and resume / 数据、采样、训练、评估、恢复。
- `scripts/`: launchers, checks and preparation / 入口、检查、准备。
- `configs/`: explicit experiment settings / 显式实验设置。
- `requirements/`: environment locks / 环境依赖锁。
- `tests/`: protocol and implementation checks / 协议和实现检查。
- `docs/`: protocols, dated status and handoffs / 协议、状态、交接。
- `third_party/mm_graph/`: licensed reference loaders / 保留上游许可的加载器。
- `artifacts/audit/original_files.json`, `lp_split_overlap.json`: portable integrity/split baselines / 随源码分发的原件与划分审计基线。

## Why launchers remain / 为什么保留多个入口

Historical launchers and modules are referenced by local frozen input manifests for resume and result verification. A file not currently used as the default can still be required by those identities. Cleanup removes temporary report helpers without changing the scientific pipeline. Tests remain to check experimental behavior.

历史入口和模块被冻结输入清单引用，断点恢复与结果核验依赖它们。不作为当前默认入口不等于无用，删除可能使续跑校验失败。本次清理临时报告脚本，保留科研流程与测试。

## Portability / 迁移限制

Data, runs, external checkpoints, workflow ledgers, frozen delivery manifests, local reports and the supplied PDF remain on the original workstation and are excluded from Git. Some entry points depend on these records and absolute-path identities and cannot run immediately on a clean clone. Recreate prerequisites for new runs using stage documentation; never alter historical identities to pretend missing experiments exist.

数据、runs、外部检查点、流程账本、冻结交付清单、本地汇报和论文留在原工作站，不上传。部分入口依赖这些记录及绝对路径身份，新克隆不能直接恢复已有实验。应依阶段说明准备新实验，不能篡改旧身份伪造完成状态。

The local recovery archive `.local_cleanup/removed_python_20261030.zip` is not published. Removed-file hashes are in `cleanup_20261030.json`.

本地恢复备份不上传；删除路径和哈希见 `cleanup_20261030.json`。
