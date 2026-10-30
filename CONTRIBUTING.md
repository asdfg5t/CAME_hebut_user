# Contributing / 贡献说明

Read `AGENTS.md`, `docs/reproduction_plan.md`, `docs/environment.md`, and the newest entries in `docs/current_status.md` before changing experiments.

- Preserve original author files and paper hashes. Add infrastructure under `src/came_repro/` and `scripts/`.
- State the objective, evidence, assumptions, budget and acceptance criteria before protocol changes. Record engineering adaptations separately.
- Select on validation, keep raw data immutable and LP topology training-only. Never overwrite historical runs or frozen releases.
- Run relevant checks and state whether they use CPU, CUDA, real or synthetic data. Planning checks are not completed experiments.
- Keep secrets, datasets, checkpoints, environments and local reports out of Git. Include provenance with derived results.

修改实验前先阅读上述文档。保护作者原件与哈希，新增实现放独立目录。协议变更说明目标、证据、假设、预算及验收条件，工程适配单列。验证集选模、原始数据不改、LP 仅用训练拓扑，历史结果与冻结版本不覆盖。运行相关检查并说明数据和计算环境，不将 dry-run 写为实验完成。凭据、数据、检查点、环境及本地报告不提交 Git，结果须附来源。
