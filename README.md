# CAME reproduction at HEBUT

[English](README.md) | [简体中文](README_zh-CN.md)

An independent reproduction harness for **A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs (CAME)**. This user-maintained project is not the authors' official implementation.

## Scope and status

The supplied `model.py` and `fusion_modules.py` are preserved byte-for-byte. Infrastructure lives in `src/came_repro/`, launchers in `scripts/`, and explicit experiment settings in `configs/`.

The latest documented status is **2026-10-22**, not a new measurement at publication time:

- Eight main tasks have documented downstream repeats: Ele-fashion, Wiki-CS, Arxiv, Goodreads-NC, FB15K237, Amazon-Sports, Amazon-Cloth and WN18RR (470 results total).
- These repeats reuse a pretrained encoder; they are not independent pretraining seeds. Some results remain below paper values.
- Goodreads-LP has separate protocol-rebuild work; the status document does not establish completion of all formal repeats.
- Products has ten completed classifier heads, but full testing paused at 427,136 / 2,213,091 rows. Final test accuracy was unavailable in the latest status.
- Baseline, few-shot and ablation entry points exist; implementation or dry-run success does not establish completed experiments.

See [current status](docs/current_status.md) for dated evidence. Local results and checkpoints are **not included in this source release**. This repository does not claim complete reproduction of the paper.

## Environment

The validated local setup is Windows, Python 3.9.13, PyTorch 2.1.2+cu118, DGL 1.1.2+cu118 and an RTX 3060 Laptop (6 GiB). This differs from the paper's server. Training requires actual **PyTorch and DGL CUDA operations**, with no automatic CPU fallback.

On the original workstation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
& .\.venv-came39\Scripts\python.exe scripts/check_environment.py
```

The bootstrap script uses `C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe`. On another workstation, create the environment with your Python 3.9 installation:

```powershell
& 'C:\path\to\Python39\python.exe' -m venv .venv-came39
& .\.venv-came39\Scripts\python.exe -m pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
& .\.venv-came39\Scripts\python.exe -m pip install -r requirements/phase_02_windows-cu118.lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
& .\.venv-came39\Scripts\python.exe -m pip install -e . --no-deps
```

## Data and running

1. Read [the plan](docs/reproduction_plan.md), [environment](docs/environment.md), and [current status](docs/current_status.md).
2. Obtain the supplied paper separately as root `CAME 4545 ICML 2026.pdf`. Its SHA256 must match `artifacts/audit/original_files.json`; the integrity guard requires it even though the PDF is not distributed here.
3. Prepare official datasets with `scripts/download_datasets.py`, `scripts/fetch_reproduction_assets.py`, and stage-specific preparation described in the plan. Keep revisions, publisher checksums and raw files unchanged. Cloning does not download data.
4. Check machine-specific paths, including `configs/storage.json`. Historical ledgers and delivery manifests are local records, not portable releases. Rebuild prerequisites on a new machine; never fabricate a completed ledger to bypass checks.
5. Select the documented entry point. `run_00_main_pipeline.py` defaults to four node tasks, not every experiment. Products uses `run_24_products_sampled_stream.py`; 25–37 cover baselines, few-shot and ablations. Their local prepared inputs are required; source alone does not restore existing workflows.

To inspect the four-node plan without starting training:

```powershell
& .\.venv-came39\Scripts\python.exe scripts/run_00_main_pipeline.py --dry-run
```

See [repository guide](docs/repository_guide.md) for navigation and portability limits. Do not launch historical experiments based only on numeric prefixes.

## Protocol safeguards

- Two-layer GAT, width 1024; six experts per view means **18 total** in the author's argument. Loss weights: `w12=0.5`, `w13=0.1`, `w23=0.5`.
- Separate paper settings from engineering adaptations. Fixed subset rounds are not full graph epochs.
- LP topology uses training edges only. Official directed splits have reverse-pair overlap; report overlap-filtered evaluation separately (see `artifacts/audit/lp_split_overlap.json`).
- Use validation for selection, never test. Preserve official Goodreads-NC exclusions and edge count (36 below the paper table).
- Resume into new runs with parent checkpoint hashes, optimizer, RNG and sampling cursor. Never overwrite historical runs.

## Checks

```powershell
& .\.venv-came39\Scripts\python.exe -m pytest tests/test_run_environment.py tests/test_paper_task_groups.py -q
```

These are focused CPU checks, not certification of CUDA training, data availability or paper accuracy. Other tests may require local data, manifests or CUDA.

## License and attribution

[LICENSE](LICENSE) grants MIT terms for original reproduction infrastructure only. Author files, the paper and third-party material are excluded; see [notices](THIRD_PARTY_NOTICES.md). MM-Graph reference code retains its upstream MIT license.

See [contribution guidance](CONTRIBUTING.md) and the [cleanup record](docs/cleanup_20261030.md).
