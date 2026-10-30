# CAME reproduction project instructions

- Read `docs/reproduction_plan.md`, `docs/environment.md` and the latest `docs/current_status.md` before changing the experiment protocol.
- The user-designated default Python is `C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe` (3.9.13). Run this project's code with `.venv-came39\Scripts\python.exe`; do not infer the default from PATH.
- Preserve root `model.py`, `fusion_modules.py` and `CAME 4545 ICML 2026.pdf` byte-for-byte. Verify `artifacts/audit/original_files.json`; add infrastructure under `src/came_repro/` and `scripts/`.
- Before each stage, state the objective, paper/code evidence, unresolved assumptions, resource budget and acceptance criteria. Revise a plan when evidence contradicts it.
- Keep paper parameters separate from engineering adaptations. Six experts per view means 18 total for the author's `num_experts` argument. Loss weights are w12=.5, w13=.1, w23=.5.
- Require actual torch AND DGL CUDA operations. Never silently fall back to CPU, reduce model width, substitute BFS for PPR, or average two modalities before the author's encoders.
- Keep raw datasets immutable, pin revisions and verify publisher SHA256. For LP, build graph topology from train edges only. Use validation for model selection, never test.
- Read `artifacts/audit/lp_split_overlap.json`: official LP splits have reverse-pair overlap. Do not add reverse edges blindly or conflate official directed evaluation with an evaluation that removes reverse overlap. Preserve the official Goodreads-NC edge count (36 below the PDF table) and official excluded node indices.
- Persist resolved configuration, code/environment/data provenance, human logger, JSONL/CSV metrics, TensorBoard events, consistent checkpoints including RNG/sampling cursor, numerical diagnostics and labeled PNG figures in a unique run directory.
- Resume into a new run with parent checkpoint hash. Do not overwrite historical experiments or claim a partial seed/step budget is a complete epoch.
- Do not call an engineering pilot a successful paper reproduction. Formal claims require the intended joint-pretraining protocol, official downstream settings, confirmed feature semantics and multiple seeds.
- Python modules/functions use snake_case, classes PascalCase. Preserve upstream asset names under `data/raw/`; name new configs and artifacts explicitly by stage/task.
- Run checks relevant to changes; verify figures visually and metrics against saved source numbers. Report missing author details and capacity limits plainly.
