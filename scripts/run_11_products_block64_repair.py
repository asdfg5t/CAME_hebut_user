"""One-shot dependency waiter and Products CUDA block-ID compatibility diagnosis.

No active source files change. CPU CSC remains compact; GPU blocks use int64.
"""
import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'scripts'))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
OUT = ROOT/'artifacts/products_block64_20261018'
LEDGER = ROOT/'artifacts/workflow_state/products_block64_20261018_current.json'
PRIOR = ROOT/'artifacts/workflow_state/all_unresolved_20261018_current.json'
PAUSE = ROOT/'artifacts/workflow_state/all_unresolved_20261018_pause.json'
PROFILE = ROOT/'configs/products_block64_20261018.json'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def block64_forward(original):
    def forward(model, data, blocks, inputs):
        return original(model, data, [block.long() for block in blocks], inputs.long())
    return forward


def audit_complete_products(run):
    """Independently check saved official outputs; no model or test rerun."""
    import csv
    import numpy as np
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from came_repro.author_loader import fingerprint
    from came_repro.prepared_data import prepared_directory
    from came_repro.filesystem_paths import windows_io_path
    run = Path(run); summary = read(run/'summary.json'); lock = read(OUT/'products_selection_lock.json')
    ids = np.load(prepared_directory('ogbn-products')/'test_indices.npy', mmap_mode='r')
    labels = np.load(prepared_directory('ogbn-products')/'labels.npy', mmap_mode='r').reshape(-1)
    with np.load(run/'test_predictions.npz') as saved:
        np.testing.assert_array_equal(saved['node_ids'], ids)
        np.testing.assert_array_equal(saved['labels'], labels[ids])
        predictions = saved['prediction']
        if predictions.shape != ids.shape or not ((predictions >= 0) & (predictions < 47)).all():
            raise ValueError('Invalid complete Products predictions')
        accuracy = float(np.mean(predictions == saved['labels']))
    if len(ids) != 2213091 or accuracy != summary['test_accuracy'] or summary['test_evaluations'] != 1:
        raise ValueError('Complete official Products accuracy differs')
    selected = Path(lock['validation_run']); manifest = read(run/'manifest.json')
    if (not lock['locked_before_test'] or lock['test_used_for_selection']
            or fingerprint(lock['classifier_checkpoint']) != lock['classifier_sha256']
            or fingerprint(selected/'summary.json') != lock['validation_summary_sha256']
            or fingerprint(OUT/'products_selection_lock.json') != summary['selection_lock_sha256']
            or manifest['parent_sha256'] != lock['classifier_sha256']):
        raise ValueError('Frozen Products validation selection or parent changed')
    for directory in (selected, run):
        rows = [json.loads(line) for line in (directory/'logs/metrics.jsonl').read_text().splitlines()]
        with (directory/'logs/metrics.csv').open(newline='') as file: csv_rows = list(csv.DictReader(file))
        if not rows or len(rows) != len(csv_rows): raise ValueError('Missing Products metric rows')
        for row, other in zip(rows, csv_rows):
            if any(not np.isfinite(value) or float(other[key]) != float(value) for key, value in row.items()):
                raise ValueError('Products CSV/JSONL mismatch')
        events = EventAccumulator(str(windows_io_path(directory/'tensorboard')), size_guidance={'scalars': 0}); events.Reload()
        for tag in rows[0]:
            values = events.Scalars(tag)
            if len(values) != len(rows) or any(event.step != row['step'] or not np.isclose(event.value, row[tag], rtol=1e-6, atol=1e-8)
                                               for event, row in zip(values, rows)):
                raise ValueError('Products TensorBoard mismatch')
    validation = read(selected/'summary.json')
    checkpoint_directory = Path(read(selected/'checkpoints/location.json')['checkpoint_directory'])
    checkpoint = torch.load(str(checkpoint_directory/'probe_last.pt'), map_location='cpu', weights_only=False)
    history = [json.loads(line) for line in (selected/'logs/metrics.jsonl').read_text().splitlines()]
    if (checkpoint['history_rows'] != history or checkpoint['next_epoch'] != len(history)+1
            or len(history) != validation['epochs_executed']
            or checkpoint['selected_epoch'] != validation['selected_epoch']
            or not all(key in checkpoint for key in ('optimizer', 'rng', 'sampling_generator', 'best_state'))):
        raise ValueError('Products complete resume state differs')
    figure = selected/'figures/probe_curves.png'
    return dict(passed=True, full_official_test_recomputed=True, official_test_nodes=len(ids), accuracy=accuracy,
                test_evaluations=1, selection_and_parent_sha_verified=True, CSV_JSONL_TensorBoard_checkpoint_verified=True,
                figure=str(figure), figure_sha256=fingerprint(figure), figure_visual_verified=False,
                formal_repeats_started=0, paper_reproduction_complete=False)


def run_stage(stage, resume=None):
    import torch
    import numpy as np
    import dgl
    from came_repro.protocol_repair import check_cuda
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import atomic_write_json
    from came_repro.train import make_model
    from came_repro.prepared_data import PreparedData
    from came_repro import stream_inference as stream
    from came_repro import experiment as experiments
    import unresolved_backend_20261018 as base
    verify_originals(); check_cuda(); torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    old_out = base.OUT; old_forward = stream.batch_full_neighbor_embeddings; old_factory = experiments.Experiment
    # Scoped runtime adaptation in this isolated GPU-lock-owning process.
    base.OUT = OUT; stream.batch_full_neighbor_embeddings = block64_forward(old_forward)
    def factory(config, audit, parent=None):
        config = copy.deepcopy(config)
        config.setdefault('evaluation', {}).update(products_GPU_block_ID_type='int64',
            products_CPU_CSC_ID_type='int32', compatibility_adapter_sha256=fingerprint(__file__),
            compatibility_profile_sha256=fingerprint(PROFILE))
        config['deviations'] = list(config.get('deviations', []))+[
            'CPU incoming CSC uses int32; sampled blocks are cast to int64 before CUDA GAT operations; all neural values remain FP32.',
            'Compatibility diagnosis after retained large-graph int32 CUDA illegal-memory failure, not a test-score optimization.']
        return old_factory(config, audit, parent)
    experiments.Experiment = factory
    try:
        if stage == 'small':
            from came_repro.compact_csc import build_index, load_graph
            saved = torch.load(str(base.encoder()), map_location='cpu', weights_only=False, mmap=True)
            model = make_model(saved['config']); model.load_state_dict(saved['model']); model.eval()
            edges = np.array([[0, 1], [0, 1], [1, 2], [2, 0], [3, 3], [4, 2]])
            directory = OUT/'fixture_csc'; build_index(edges, 7, directory, 'fixture')
            graph = load_graph(directory); features = np.random.default_rng(812).normal(size=(7, 1024)).astype(np.float32)
            data = PreparedData(graph, features, None, {}, {})
            with torch.no_grad():
                raw = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=7).to('cuda')
                expected = model(raw, torch.from_numpy(features[:, :512]).cuda(), torch.from_numpy(features[:, 512:]).cuda(),
                    getz3=True, compute_loss=False).cpu().numpy()
                inputs, targets, blocks = dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(graph, torch.arange(7, dtype=torch.int32))
                if any(block.long().idtype != torch.int64 for block in blocks): raise ValueError('GPU block cast failed')
                actual = stream.batch_full_neighbor_embeddings(model, data, blocks, inputs).cpu().numpy()
            if not np.isfinite(actual).all() or not np.isfinite(expected).all(): raise FloatingPointError('Nonfinite fixture')
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            atomic_write_json(OUT/'small_acceptance.json', dict(passed=True, actual_Torch_DGL_CUDA=True,
                GPU_blocks='int64', CPU_graph='int32_CSC', feature_precision='fp32', max_abs_difference=float(np.max(np.abs(actual-expected))),
                adapter_sha256=fingerprint(__file__), base_profile_sha256=fingerprint(base.PROFILE),
                compatibility_profile_sha256=fingerprint(PROFILE), fixture_not_benchmark=True))
        else:
            receipt = read(OUT/'small_acceptance.json')
            if (not receipt['passed'] or receipt['adapter_sha256'] != fingerprint(__file__)
                    or receipt['base_profile_sha256'] != fingerprint(base.PROFILE)
                    or receipt['compatibility_profile_sha256'] != fingerprint(PROFILE)):
                raise ValueError('Current compatibility CUDA acceptance required')
            if stage == 'calibration': base.products_calibration()
            else:
                from unresolved_products_full_20261018 import main as full
                full(resume)
    finally:
        base.OUT = old_out; stream.batch_full_neighbor_embeddings = old_forward; experiments.Experiment = old_factory


def pipeline():
    import psutil
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    OUT.mkdir(parents=True, exist_ok=True)
    prior = read(PRIOR); sources = dict(prior['identity']['sources']); sources[str(Path(__file__))] = fingerprint(__file__)
    identity = dict(sources=sources, base_profile_sha256=prior['identity']['profile_sha256'], compatibility_profile_sha256=fingerprint(PROFILE))
    guard = acquire_gpu_lock(ROOT/'.runtime/products_block64_repair.lock')
    sleep_guard = False
    if os.name == 'nt':
        import ctypes
        sleep_guard = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    ledger = read(LEDGER) if LEDGER.exists() else dict(identity=identity, state='waiting_for_prior_serial_queue', records={}, active_seconds=0.)
    if ledger['identity'] != identity: guard.close(); raise ValueError('Compatibility identity changed')
    atomic_write_json(LEDGER, ledger)
    try:
        while not read(PRIOR)['state'].startswith('completed'):
            current = read(PRIOR)
            if PAUSE.exists() or current['state'] in ('paused', 'failed'):
                ledger['state'] = 'deferred_prior_pause_or_failure'; atomic_write_json(LEDGER, ledger); return
            launch = read(ROOT/'artifacts/all_unresolved_20261018/launch.json')
            try:
                process = psutil.Process(launch['pid'])
                alive = abs(process.create_time()-launch['creation_time']) < 1
            except psutil.NoSuchProcess: alive = False
            if not alive:
                ledger['state'] = 'deferred_prior_external_exit_requires_review'; atomic_write_json(LEDGER, ledger); return
            time.sleep(15)
        # The original queue may be writing its last receipt; never overlap GPU work.
        completed = read(PRIOR)
        if completed.get('results', {}).get('products') or completed.get('records', {}).get('products_full', {}).get('state') == 'completed':
            ledger['state'] = 'completed_products_already_evaluated_no_retest'; atomic_write_json(LEDGER, ledger); return
        verify_originals()
        if (fingerprint(ROOT/'configs/all_unresolved_20261018.json') != identity['base_profile_sha256']
                or fingerprint(PROFILE) != identity['compatibility_profile_sha256']): raise ValueError('Frozen compatibility plan changed')
        for path, sha in sources.items():
            if fingerprint(path) != sha: raise ValueError('Frozen source changed: '+path)
        from came_repro.quality_pipeline import run_child, best_resume
        from came_repro.storage_budget import check_budget
        from came_repro.pause_control import TrainingPaused
        from cloth_wn_autonomous import artifact_hashes
        import logging
        logger = logging.getLogger('products_block64'); logger.setLevel(logging.INFO)
        handler = logging.FileHandler(OUT/'pipeline.log', encoding='utf-8'); stream_handler = logging.StreamHandler()
        for h in (handler, stream_handler):
            fmt = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s'); fmt.converter = time.gmtime; h.setFormatter(fmt); logger.addHandler(h)
        probe = acquire_gpu_lock(); probe.close(); check_budget(additional_project_bytes=64*2**20, additional_cache_bytes=64*2**20)
        def execute(name, checkpoint=None):
            if PAUSE.exists(): raise TrainingPaused('Explicit task pause retained')
            if (fingerprint(ROOT/'configs/all_unresolved_20261018.json') != identity['base_profile_sha256']
                    or fingerprint(PROFILE) != identity['compatibility_profile_sha256']): raise ValueError('Frozen profile changed')
            for path, sha in sources.items():
                if fingerprint(path) != sha: raise ValueError('Frozen source changed')
            record = ledger['records'].setdefault(name, dict(state='ready', attempts=[]))
            if record['state'] == 'completed':
                for path, sha in record['artifacts'].items():
                    if fingerprint(path) != sha: raise ValueError('Completed artifact changed')
                return record
            args = [str(Path(__file__)), '--stage', name]
            parent = best_resume(record, checkpoint) if checkpoint else None
            if parent: args += ['--resume-head', str(parent)]
            remaining = 24*3600-read(PRIOR)['active_seconds']-ledger['active_seconds']
            if remaining <= 0: raise TrainingPaused('Combined finite activity budget exhausted')
            ledger['state'] = 'running_'+name; atomic_write_json(LEDGER, ledger)
            code, run = run_child(args, PAUSE, ledger, LEDGER, record, logger, remaining)
            if code == 75: raise TrainingPaused('Durable Products cursor/head saved')
            if code: raise RuntimeError('Compatibility diagnostic failed: '+str(code))
            summary = read(OUT/'small_acceptance.json') if name == 'small' else read(Path(run)/'summary.json')
            record.update(state='completed', run_directory=run, summary=summary,
                artifacts={str(OUT/'small_acceptance.json'): fingerprint(OUT/'small_acceptance.json')} if name == 'small' else artifact_hashes(run))
            atomic_write_json(LEDGER, ledger); return record
        execute('small'); calibration = execute('calibration')
        if calibration['summary']['ready_for_full_start']:
            complete = execute('full', 'probe_last.pt')
            ledger['result'] = audit_complete_products(complete['run_directory'])
            atomic_write_json(OUT/'independent_full_test_audit.json', ledger['result'])
            ledger['state'] = 'completed_full_products_pending_visual_review'
        else: ledger['state'] = 'completed_compatibility_calibration_capacity_insufficient'
        atomic_write_json(LEDGER, ledger)
    except BaseException as error:
        ledger.update(state='paused' if PAUSE.exists() or type(error).__name__ == 'TrainingPaused' else 'failed', error=str(error))
        atomic_write_json(LEDGER, ledger); raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        guard.close()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage', choices=['small', 'calibration', 'full'])
    parser.add_argument('--resume-head', type=Path); parser.add_argument('--detach', action='store_true'); parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.stage: run_stage(args.stage, args.resume_head); return
    if not args.detach: pipeline(); return
    import psutil
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.atomic_io import atomic_write_json
    OUT.mkdir(parents=True, exist_ok=True); guard = acquire_gpu_lock(ROOT/'.runtime/products_block64_launch.lock')
    try:
        file = OUT/'launch.json'
        if file.exists():
            old = read(file)
            try:
                p = psutil.Process(old['pid'])
                if abs(p.create_time()-old['creation_time']) < 1: print(json.dumps(dict(already_running=True, **old))); return
            except psutil.NoSuchProcess: pass
        if PAUSE.exists(): raise RuntimeError('Explicit task pause retained')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); log = OUT/('console_'+stamp+'.log')
        with log.open('ab') as output:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'], cwd=str(ROOT), stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, close_fds=True,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1'})
        value = dict(pid=child.pid, creation_time=psutil.Process(child.pid).create_time(), console_log=str(log),
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), detached_breakaway_accepted=True, one_shot_waiter_not_recurring_automation=True)
        atomic_write_json(file, value); print(json.dumps(value))
    finally: guard.close()


if __name__ == '__main__': main()
