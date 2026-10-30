"""One finite Cloth ranking-objective diagnosis, with exact resume and offline launch."""
import argparse
import copy
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
PROFILE = ROOT/'configs/cloth_pairwise_20261017.json'
OUT = ROOT/'artifacts/cloth_pairwise_20261017'
LEDGER = ROOT/'artifacts/workflow_state/cloth_pairwise_current.json'
PAUSE = ROOT/'artifacts/workflow_state/cloth_pairwise_pause.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_plan(plan):
    if (plan['schema'] != 'came_cloth_pairwise_validation_v1' or plan['seed'] != 0
            or plan['only_task'] != 'cloth-copurchase' or plan['candidate_count'] != 1
            or plan['formal_repeats'] or plan['new_test_limit'] != 1
            or plan['minimum_validation_gain_pp'] != .5 or plan['maximum_active_hours'] != 8):
        raise ValueError('Fixed single-candidate validation protocol changed')
    expected = dict(head_variant='two_hidden_layernorm_head_upstream_schedule', lr=.001,
                    batch_size=4096, max_epochs=300, patience=20, step_lr_size=None,
                    step_lr_gamma=None, loss_reduction='pairwise_softplus_mean')
    if plan['recipe'] != expected or plan['negative_pool_size'] != 4:
        raise ValueError('Declared pairwise objective, head or resource budget changed')


def passes_gate(plan, value):
    import math
    if not math.isfinite(value):
        raise ValueError('Nonfinite validation MRR')
    return value-plan['reference_validation_mrr'] >= plan['minimum_validation_gain_pp']/100-1e-12


def acceptance():
    import numpy as np
    import torch
    from came_repro.protocol_repair import check_cuda
    from came_repro.experiment import Experiment, seed_everything
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.cloth_pairwise import train_pairwise, pairwise_loss
    from came_repro.pause_control import TrainingPaused
    import came_repro.pause_control as pauses
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from cloth_wn_autonomous_backend import assert_same
    check_cuda(); torch.set_num_threads(4); seed_everything(713)
    # These are synthetic fixtures, never reported as benchmark accuracy.
    gpu = torch.randn(16, 1024, device='cuda')
    source = np.arange(6)
    train = dict(source_node=source, target_node=source+1)
    valid = dict(source_node=np.array([8, 9]), target_node=np.array([9, 10]),
                 target_node_neg=np.tile(np.arange(16), (2, 10))[:, :150])
    recipe = copy.deepcopy(read(PROFILE)['recipe']); recipe.update(batch_size=3, max_epochs=3)
    config = dict(name='cloth_pairwise_resume_fixture', seed=17,
                  storage=read(ROOT/'configs/storage.json'), evaluation=dict(fixture_only=True, recipe=recipe))
    audit = dict(fixture_only_not_benchmark=True)
    reference = Experiment(config, audit)
    try:
        with use_cached_vectors(): train_pairwise(gpu, train, valid, recipe, 17, reference)
        reference.status('completed_pairwise_fixture_reference')
    finally: reference.close()
    original = pauses.pause_requested; calls = [0]
    def pause_after_two():
        calls[0] += 1
        return calls[0] >= 2
    pauses.pause_requested = pause_after_two
    paused = Experiment(config, audit)
    try:
        try:
            with use_cached_vectors(): train_pairwise(gpu, train, valid, recipe, 17, paused)
        except TrainingPaused: paused.status('paused_pairwise_fixture')
        else: raise ValueError('Synthetic pause was not triggered')
    finally: paused.close(); pauses.pause_requested = original
    checkpoint = paused.checkpoint_directory/'link_head_last.pt'
    resumed = Experiment(config, audit, checkpoint)
    try:
        with use_cached_vectors(): train_pairwise(gpu, train, valid, recipe, 17, resumed, checkpoint)
        resumed.status('completed_pairwise_fixture_resume')
    finally: resumed.close()
    a = torch.load(str(reference.checkpoint_directory/'link_head_last.pt'), map_location='cpu', weights_only=False)
    b = torch.load(str(resumed.checkpoint_directory/'link_head_last.pt'), map_location='cpu', weights_only=False)
    for key in ('head', 'optimizer', 'rng', 'sampling_generator', 'history_rows', 'best_state', 'selected_epoch', 'next_epoch'):
        assert_same(a[key], b[key])
    p = torch.tensor([1., 0., -2.], device='cuda', requires_grad=True)
    n = torch.tensor([0., 2., -1.], device='cuda', requires_grad=True)
    loss = pairwise_loss(p, n); loss.backward()
    torch.testing.assert_close(loss, (-torch.nn.functional.logsigmoid(p-n)).mean())
    if not bool((p.grad < 0).all() and (n.grad > 0).all()):
        raise ValueError('Pairwise gradient signs incorrect')
    atomic_write_json(OUT/'acceptance.json', dict(passed=True, actual_Torch_and_DGL_CUDA=True,
        pairwise_logistic_identity_and_gradient_verified=True, full_optimizer_RNG_sampling_history_bitexact=True,
        fixture_only_not_benchmark=True, reference_run=str(reference.directory), paused_run=str(paused.directory),
        resumed_run=str(resumed.directory), parent_sha256=fingerprint(checkpoint),
        profile_sha256=fingerprint(PROFILE), script_sha256=fingerprint(__file__),
        trainer_sha256=fingerprint(ROOT/'src/came_repro/cloth_pairwise.py')))


def validate(resume=None):
    import numpy as np
    import torch
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import atomic_write_json
    from came_repro.protocol_repair import check_cuda
    from came_repro.lp_validation_repair import load_verified_cache
    from came_repro.experiment import Experiment
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.cloth_pairwise import train_pairwise
    from came_repro.pause_control import TrainingPaused
    plan = read(PROFILE); validate_plan(plan); verify_originals(); check_cuda(); torch.set_num_threads(4)
    accepted = read(OUT/'acceptance.json')
    if (not accepted['passed'] or accepted['profile_sha256'] != fingerprint(PROFILE)
            or accepted['script_sha256'] != fingerprint(__file__)
            or accepted['trainer_sha256'] != fingerprint(ROOT/'src/came_repro/cloth_pairwise.py')):
        raise ValueError('Matching actual CUDA/resume acceptance required')
    reference = ROOT/plan['reference_run']; summary = read(reference/'summary.json')
    digest = plan['encoder_sha256']; cache = Path(summary['embedding_cache'])
    gpu, audit = load_verified_cache('cloth-copurchase', cache, digest)
    if audit.get('diagnostic_topology', {}).get('policy') != 'upstream_train_bidirected_unique':
        raise ValueError('Frozen complete train-only topology required')
    raw = ROOT/'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
    if fingerprint(raw) != audit['raw_sha256']['lp-edge-split.pt']:
        raise ValueError('Official immutable split changed')
    split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
    if (len(gpu) != 125839 or len(split['valid']['source_node']) != 121323
            or len(split['test']['source_node']) != 97456
            or np.shape(split['valid']['target_node_neg'])[1] != 150):
        raise ValueError('Complete official Cloth data required')
    config = read(reference/'config.json'); config.update(name='cloth_pairwise_ranking', seed=0)
    recipe = plan['recipe']
    config['evaluation'].update(recipe, objective='pairwise_softplus_mean', embedding_cache=str(cache),
        encoder_sha256=digest, pairwise_profile_sha256=fingerprint(PROFILE), test_used_for_selection=False)
    config['deviations'] = list(config.get('deviations', [])) + [
        'BPR pairwise logistic objective replaces reconstructed BCE; this is NOT confirmed CAME author LP loss.',
        'One fixed seed0 fresh head, frozen original encoder and complete graph/splits, existing K4 train-only hard negatives.',
        'Historical test scores were previously seen; new candidate selection uses validation only.']
    encoder = Path(read(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(encoder) != digest: raise ValueError('Protected encoder identity changed')
    experiment = Experiment(config, audit, resume or encoder)
    try:
        with use_cached_vectors(): result = train_pairwise(gpu, split['train'], split['valid'], recipe, 0, experiment, resume)
        result.update(dataset='cloth-copurchase', task='link', representation='came', encoder_sha256=digest,
            embedding_cache=str(cache), graph_policy='upstream_train_bidirected_unique',
            objective='pairwise_softplus_mean', diagnostic_only=True, paper_reproduction_complete=False)
        atomic_write_json(experiment.directory/'summary.json', result)
        experiment.status('completed_cloth_pairwise_validation_only')
    except BaseException as error:
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally: experiment.close()


def pipeline():
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child, best_resume
    from came_repro.storage_budget import check_budget
    from came_repro.pause_control import TrainingPaused
    from cloth_wn_autonomous import artifact_hashes
    plan = read(PROFILE); validate_plan(plan); verify_originals()
    if PAUSE.exists(): raise TrainingPaused('Explicit pause retained; pass --resume to continue')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    source_paths = list(read(ROOT/'artifacts/workflow_state/cloth_wn_autonomous_current.json')['identity']['sources'])
    source_paths += [str(Path(__file__)), str(ROOT/'src/came_repro/cloth_pairwise.py')]
    identity = dict(sources={p: fingerprint(p) for p in source_paths}, profile_sha256=fingerprint(PROFILE))
    guard = acquire_gpu_lock(ROOT/'.runtime/cloth_pairwise.lock')
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = read(LEDGER) if LEDGER.exists() else dict(identity=identity, records={}, active_seconds=0., state='prepared')
    if ledger['identity'] != identity: guard.close(); raise ValueError('Frozen candidate identity changed')
    logger = logging.getLogger('cloth_pairwise'); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(OUT/'pipeline.log', encoding='utf-8'); stream = logging.StreamHandler()
    import time
    for h in (handler, stream):
        formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s'); formatter.converter = time.gmtime
        h.setFormatter(formatter); logger.addHandler(h)
    sleep_guard = False
    if os.name == 'nt':
        import ctypes
        sleep_guard = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    def persist(): atomic_write_json(LEDGER, ledger)
    def execute(name, arguments, checkpoint_name=None):
        verify_originals()
        if fingerprint(PROFILE) != identity['profile_sha256']: raise ValueError('Plan changed')
        for p, sha in identity['sources'].items():
            if fingerprint(p) != sha: raise ValueError('Source changed: '+p)
        if PAUSE.exists(): raise TrainingPaused('Pause requested')
        record = ledger['records'].setdefault(name, dict(state='ready', attempts=[]))
        if record['state'] == 'completed':
            for p, sha in record['artifacts'].items():
                if fingerprint(p) != sha: raise ValueError('Completed artifact changed')
            return record
        if name == 'confirmation':
            for attempt in record['attempts']:
                if attempt.get('run_directory') and (Path(attempt['run_directory'])/'test_link_ranks.npz').exists():
                    raise ValueError('Interrupted test output exists; refuse automatic retest')
        parent = best_resume(record, checkpoint_name) if checkpoint_name else None
        if parent: arguments += ['--resume-head', str(parent)]
        remaining = plan['maximum_active_hours']*3600-ledger['active_seconds']
        if remaining <= 0: raise TrainingPaused('Finite active budget exhausted')
        ledger['state'] = 'running_'+name; persist()
        code, run = run_child(arguments, PAUSE, ledger, LEDGER, record, logger, remaining)
        if code == 75 or PAUSE.exists(): raise TrainingPaused('Cooperative pause saved')
        if code: raise RuntimeError('GPU child failed; do not mask runtime/data error: '+str(code))
        if name == 'acceptance':
            summary = read(OUT/'acceptance.json'); assert summary['passed']
            artifacts = {str(OUT/'acceptance.json'): fingerprint(OUT/'acceptance.json')}
        else:
            summary = read(Path(run)/'summary.json'); artifacts = artifact_hashes(run)
        record.update(state='completed', run_directory=run, summary=summary, artifacts=artifacts); persist()
        return record
    try:
        check_budget(**plan['budget']); probe = acquire_gpu_lock(); probe.close()
        execute('acceptance', [str(Path(__file__)), '--stage', 'acceptance'])
        record = execute('validation', [str(Path(__file__)), '--stage', 'validation'], 'link_head_last.pt')
        result = record['summary']; score = result['validation_mrr']
        ledger['validation_gain_pp'] = 100*(score-plan['reference_validation_mrr'])
        ledger['passes_validation_gate'] = passes_gate(plan, score); persist()
        if not ledger['passes_validation_gate']:
            ledger.update(state='completed_validation_rejected', new_tests=0, old_test_retained=True)
        else:
            selected = Path(record['run_directory']); cache = Path(result['embedding_cache'])
            head = Path(read(selected/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
            lock = dict(dataset='cloth-copurchase', encoder_sha256=plan['encoder_sha256'], classifier_checkpoint=str(head),
                classifier_sha256=fingerprint(head), validation_run_directory=str(selected),
                validation_summary_sha256=fingerprint(selected/'summary.json'), embedding_cache=str(cache),
                inference_state_sha256=fingerprint(cache/'inference_state.json'), locked_before_test=True,
                test_used_for_selection=False, head_variant=result['head_variant'], selected_validation_mrr=score)
            lock_path = OUT/'selection_lock.json'
            if lock_path.exists() and read(lock_path) != lock: raise ValueError('Selection lock changed')
            atomic_write_json(lock_path, lock)
            confirmed = execute('confirmation', ['-m', 'came_repro.protocol_repair', '--stage', 'confirm_lp', '--selection-lock', str(lock_path)])
            from audit_two_lp_confirmation import audit_confirmation
            ledger['result'] = audit_confirmation(Path(confirmed['run_directory']), lock_path)
            ledger.update(state='completed_confirmation_pending_visual_review', new_tests=1)
        persist(); atomic_write_json(OUT/'outcome.json', ledger)
        logger.info('PAIRWISE_QUEUE_COMPLETED state=%s gain_pp=%.8f', ledger['state'], ledger['validation_gain_pp'])
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); persist(); raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for h in (handler, stream): h.close(); logger.removeHandler(h)
        guard.close()


def detach():
    import psutil
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.atomic_io import atomic_write_json
    OUT.mkdir(parents=True, exist_ok=True)
    guard = acquire_gpu_lock(ROOT/'.runtime/cloth_pairwise_launch.lock')
    try:
        path = OUT/'launch.json'
        if path.exists():
            previous = read(path)
            try:
                p = psutil.Process(previous['pid'])
                if abs(p.create_time()-previous['creation_time']) < 1:
                    print(json.dumps(dict(already_running=True, **previous))); return
            except psutil.NoSuchProcess: pass
        if PAUSE.exists(): raise RuntimeError('Pause retained; use explicit --resume')
        for name in ('pycharm_gpu', 'representation_coverage', 'cloth_pairwise'):
            probe = acquire_gpu_lock(ROOT/'.runtime'/(name+'.lock')); probe.close()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        log = OUT/('offline_console_'+stamp+'.log')
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
        with log.open('ab') as output:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'], cwd=str(ROOT),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True,
                env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1',
                     'TRANSFORMERS_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1'})
        value = dict(pid=child.pid, creation_time=psutil.Process(child.pid).create_time(), console_log=str(log),
            detached_breakaway_accepted=True, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_write_json(path, value); print(json.dumps(value))
    finally: guard.close()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['acceptance', 'validation'])
    parser.add_argument('--resume-head', type=Path)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from came_repro.atomic_io import atomic_write_json
    if args.pause:
        atomic_write_json(PAUSE, dict(reason='explicit_user_pause')); return
    if args.resume and PAUSE.exists():
        from came_repro.ide_workflow import acquire_gpu_lock
        guard = acquire_gpu_lock(ROOT/'.runtime/cloth_pairwise.lock')
        try:
            directory = OUT/'pause_history'; directory.mkdir(parents=True, exist_ok=True)
            PAUSE.replace(directory/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'))
        finally: guard.close()
    if args.stage == 'acceptance': acceptance()
    elif args.stage == 'validation': validate(args.resume_head)
    elif args.detach: detach()
    else: pipeline()


if __name__ == '__main__': main()
