"""Offline serial queue for four unresolved main tasks; no formal seed grid."""
import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'scripts'))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
PROFILE = ROOT/'configs/all_unresolved_20261018.json'
OUT = ROOT/'artifacts/all_unresolved_20261018'
LEDGER = ROOT/'artifacts/workflow_state/all_unresolved_20261018_current.json'
PAUSE = ROOT/'artifacts/workflow_state/all_unresolved_20261018_pause.json'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_plan(p):
    if (p['schema'] != 'came_all_unresolved_v20261018' or p['seed'] != 0 or p['formal_repeats']
            or p['maximum_new_tests_per_task'] != 1 or p['validation_gain_threshold_pp'] != .5
            or p['maximum_active_hours'] != 24 or not p['offline']): raise ValueError('Fixed finite protocol changed')
    if (p['cloth']['recipe']['head_training_dropout'] != .3 or p['cloth']['recipe']['max_epochs'] != 300
            or p['wn']['lr'] != .05 or p['wn']['max_epochs'] != 2000): raise ValueError('Declared one-candidate budgets changed')
    if (p['products']['width'] != 1024 or p['products']['nodes'] != 2449029
            or p['products']['official_test_nodes'] != 2213091): raise ValueError('Official full data/width required')


def passes_gate(p, task, summary):
    import math
    if summary['test_evaluations'] != 0 or summary['seed'] != 0: raise ValueError('Validation-only seed0 required')
    value = summary['validation_mrr' if task == 'cloth' else 'validation_accuracy']
    if not math.isfinite(value): raise ValueError('Nonfinite validation')
    return value-p[task]['reference_validation'] >= p['validation_gain_threshold_pp']/100-1e-12


def pipeline():
    import time
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child, best_resume
    from came_repro.storage_budget import check_budget
    from came_repro.pause_control import TrainingPaused
    from cloth_wn_autonomous import artifact_hashes, independent_wn
    from audit_unresolved_validation import audit_validation, books_protocol_audit
    p = read(PROFILE); validate_plan(p); verify_originals()
    if PAUSE.exists(): raise TrainingPaused('Explicit queue pause retained')
    previous = read(ROOT/'artifacts/workflow_state/cloth_pairwise_current.json')
    if previous['state'] != 'completed_validation_rejected': raise ValueError('Finish previous Cloth first')
    for path, sha in previous['identity']['sources'].items():
        if fingerprint(path) != sha: raise ValueError('Previously frozen source changed')
    paths = list(previous['identity']['sources'])+[str(x) for x in (ROOT/'src/came_repro').glob('*.py')]
    paths += [str(ROOT/'scripts'/name) for name in ('run_10_unresolved_main.py', 'unresolved_backend_20261018.py',
        'unresolved_products_full_20261018.py', 'audit_unresolved_validation.py', 'run_lp_hard_negative_candidate.py',
        'run_products_single_validation.py', 'audit_two_lp_confirmation.py', 'cloth_wn_autonomous.py', 'cloth_wn_autonomous_backend.py')]
    identity = dict(sources={path: fingerprint(path) for path in paths}, profile_sha256=fingerprint(PROFILE),
                    storage_sha256=fingerprint(ROOT/'configs/storage.json'))
    guard = acquire_gpu_lock(ROOT/'.runtime/all_unresolved_20261018.lock'); OUT.mkdir(parents=True, exist_ok=True)
    ledger = read(LEDGER) if LEDGER.exists() else dict(identity=identity, state='prepared', records={}, active_seconds=0., results={}, rejections={})
    if ledger['identity'] != identity: guard.close(); raise ValueError('Queue frozen identity changed')
    logger = logging.getLogger('all_unresolved'); logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler(), logging.FileHandler(OUT/'pipeline.log', encoding='utf-8')]
    for h in handlers:
        fmt = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s'); fmt.converter = time.gmtime; h.setFormatter(fmt); logger.addHandler(h)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    sleep_guard = False
    if os.name == 'nt':
        import ctypes
        sleep_guard = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    def persist(): atomic_write_json(LEDGER, ledger)
    def execute(name, arguments, checkpoint=None, fatal=False):
        verify_originals()
        if PAUSE.exists(): raise TrainingPaused('Queue paused')
        if fingerprint(PROFILE) != identity['profile_sha256'] or fingerprint(ROOT/'configs/storage.json') != identity['storage_sha256']:
            raise ValueError('Frozen profile/storage changed')
        for path, sha in identity['sources'].items():
            if fingerprint(path) != sha: raise ValueError('Active source changed: '+path)
        r = ledger['records'].setdefault(name, dict(state='ready', attempts=[]))
        if r['state'] == 'completed':
            for path, sha in r['artifacts'].items():
                if fingerprint(path) != sha: raise ValueError('Completed artifact changed')
            return r
        if r['state'] == 'failed': return None
        if name.startswith('confirm_'):
            for attempt in r['attempts']:
                run = attempt.get('run_directory')
                if run and any((Path(run)/file).exists() for file in ('test_link_ranks.npz', 'test_predictions.npz')):
                    raise ValueError('Interrupted test output; manual audit required, no automatic retest')
        parent = best_resume(r, checkpoint) if checkpoint else None
        if parent: arguments += ['--resume-head', str(parent)]
        remaining = p['maximum_active_hours']*3600-ledger['active_seconds']
        if remaining <= 0: raise TrainingPaused('Finite queue budget exhausted')
        ledger['state'] = 'running_'+name; persist(); logger.info('STAGE %s parent=%s', name, parent)
        code, run = run_child(arguments, PAUSE, ledger, LEDGER, r, logger, remaining)
        if code == 75 or PAUSE.exists(): raise TrainingPaused('Durable cooperative stage boundary saved')
        if code:
            r.update(state='failed', exit_code=code); persist()
            if fatal: raise RuntimeError('Mandatory acceptance failed; no scientific candidates started')
            logger.error('Task %s failed with code%d; retained, other independent tasks may continue', name, code); return None
        if name == 'acceptance':
            summary = read(OUT/'acceptance.json')
            if (not summary['passed'] or summary['backend_sha256'] != fingerprint(ROOT/'scripts/unresolved_backend_20261018.py')
                    or summary['profile_sha256'] != fingerprint(PROFILE)): raise ValueError('Acceptance identity mismatch')
            artifacts = {str(OUT/'acceptance.json'): fingerprint(OUT/'acceptance.json')}
        else:
            if not run: raise ValueError('Missing unique child run')
            summary = read(Path(run)/'summary.json'); artifacts = artifact_hashes(run)
        r.update(state='completed', run_directory=run, summary=summary, artifacts=artifacts); persist(); return r
    try:
        probe = acquire_gpu_lock(); probe.close(); check_budget(**p['budget'])
        if 'previous_pairwise_audit' not in ledger:
            ledger['previous_pairwise_audit'] = audit_validation(previous['records']['validation']['run_directory'], 'cloth')
            atomic_write_json(OUT/'previous_pairwise_rejection_audit.json', ledger['previous_pairwise_audit'])
        if 'books_protocol_review' not in ledger:
            ledger['books_protocol_review'] = books_protocol_audit()
            atomic_write_json(OUT/'books_protocol_review.json', ledger['books_protocol_review'])
        persist()
        backend = str(ROOT/'scripts/unresolved_backend_20261018.py')
        execute('acceptance', [backend, '--stage', 'acceptance'], fatal=True)
        calibration = execute('products_calibration', [backend, '--stage', 'products_calibration'])
        # Both fresh accuracy candidates remain predeclared, validation-only.
        for task in ('wn', 'cloth'):
            record = execute('validate_'+task, [backend, '--stage', task],
                             'probe_last.pt' if task == 'wn' else 'link_head_last.pt')
            if record is None: continue
            summary = record['summary']; audit = audit_validation(record['run_directory'], task)
            atomic_write_json(OUT/(task+'_validation_audit.json'), audit)
            if not passes_gate(p, task, summary):
                ledger['rejections'][task] = dict(validation_score=audit['validation_score'],
                    gain_pp=100*(audit['validation_score']-p[task]['reference_validation']), new_test_evaluations=0,
                    old_test_retained=True); persist(); continue
            selected = Path(record['run_directory']); config = read(selected/'config.json')
            cache = Path(summary.get('embedding_cache') or config['evaluation']['embedding_cache'])
            directory = Path(read(selected/'checkpoints/location.json')['checkpoint_directory'])
            head = directory/('link_head_best.pt' if task == 'cloth' else 'probe_best_raw_linear.pt')
            lock = dict(dataset=summary['dataset'], encoder_sha256=p['encoder_sha256'], classifier_checkpoint=str(head),
                classifier_sha256=fingerprint(head), validation_run_directory=str(selected),
                validation_summary_sha256=fingerprint(selected/'summary.json'), embedding_cache=str(cache),
                inference_state_sha256=fingerprint(cache/'inference_state.json'), locked_before_test=True,
                test_used_for_selection=False, graph_policy=summary['graph_policy'])
            if task == 'cloth': lock.update(head_variant=summary['head_variant'], selected_validation_mrr=summary['validation_mrr'])
            else: lock.update(feature_transform='train_pair_affine', selected_validation_accuracy=summary['validation_accuracy'])
            lock_path = OUT/(task+'_selection_lock.json')
            if lock_path.exists() and read(lock_path) != lock: raise ValueError('Selection lock changed')
            atomic_write_json(lock_path, lock)
            args = (['-m', 'came_repro.protocol_repair', '--stage', 'confirm_lp', '--selection-lock', str(lock_path)] if task == 'cloth'
                    else ['-m', 'came_repro.kg_selected_confirmation', '--selection-lock', str(lock_path)])
            confirmed = execute('confirm_'+task, args)
            if confirmed:
                if task == 'cloth':
                    from audit_two_lp_confirmation import audit_confirmation
                    result = audit_confirmation(Path(confirmed['run_directory']), lock_path)
                else: result = independent_wn(Path(confirmed['run_directory']), lock_path)
                ledger['results'][task] = result; persist()
        if calibration and calibration['summary']['ready_for_full_start']:
            record = execute('products_full', [backend, '--stage', 'products_full'], 'probe_last.pt')
            if record:
                from came_repro.prepared_data import prepared_directory
                import numpy as np
                run = Path(record['run_directory']); summary = record['summary']
                ids = np.load(prepared_directory('ogbn-products')/'test_indices.npy', mmap_mode='r')
                labels = np.load(prepared_directory('ogbn-products')/'labels.npy', mmap_mode='r').reshape(-1)
                with np.load(run/'test_predictions.npz') as saved:
                    np.testing.assert_array_equal(saved['node_ids'], ids); np.testing.assert_array_equal(saved['labels'], labels[ids])
                    value = float(np.mean(saved['prediction'] == saved['labels']))
                if len(ids) != 2213091 or value != summary['test_accuracy']: raise ValueError('Independent full Products test differs')
                ledger['results']['products'] = dict(full_official_test_recomputed=True, accuracy=value, test_evaluations=1,
                    formal_repeats_started=0, figure_visual_review_pending=True); persist()
        else: ledger['products_deferred'] = 'Capacity/numerical/throughput acceptance not passed; old prefix preserved'
        ledger['state'] = 'completed_bounded_tasks_pending_manual_visual_review'
        ledger['books_author_protocol_unresolved'] = True; persist(); atomic_write_json(OUT/'outcome.json', ledger)
        logger.info('FINITE_QUEUE_COMPLETE results=%s rejected=%s; no test-driven expansion', list(ledger['results']), list(ledger['rejections']))
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); persist(); raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for h in handlers: h.close(); logger.removeHandler(h)
        guard.close()


def detach():
    import psutil
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.atomic_io import atomic_write_json
    OUT.mkdir(parents=True, exist_ok=True); guard = acquire_gpu_lock(ROOT/'.runtime/all_unresolved_20261018_launch.lock')
    try:
        file = OUT/'launch.json'
        if file.exists():
            previous = read(file)
            try:
                process = psutil.Process(previous['pid'])
                if abs(process.create_time()-previous['creation_time']) < 1: print(json.dumps(dict(already_running=True, **previous))); return
            except psutil.NoSuchProcess: pass
        if PAUSE.exists(): raise RuntimeError('Explicit queue pause retained')
        for name in ('pycharm_gpu', 'cloth_pairwise', 'all_unresolved_20261018'):
            probe = acquire_gpu_lock(ROOT/'.runtime'/(name+'.lock')); probe.close()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); log = OUT/('offline_console_'+stamp+'.log')
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
        with log.open('ab') as output:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'], cwd=str(ROOT),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True,
                env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1'})
        value = dict(pid=child.pid, creation_time=psutil.Process(child.pid).create_time(), console_log=str(log),
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), detached_breakaway_accepted=True)
        atomic_write_json(file, value); print(json.dumps(value))
    finally: guard.close()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--pause', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from came_repro.atomic_io import atomic_write_json
    if args.pause: atomic_write_json(PAUSE, dict(reason='explicit_user_pause')); return
    if args.resume and PAUSE.exists():
        from came_repro.ide_workflow import acquire_gpu_lock
        guard = acquire_gpu_lock(ROOT/'.runtime/all_unresolved_20261018.lock')
        try:
            directory = OUT/'pause_history'; directory.mkdir(parents=True, exist_ok=True)
            PAUSE.replace(directory/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'))
        finally: guard.close()
    if args.detach: detach()
    else: pipeline()


if __name__ == '__main__': main()
