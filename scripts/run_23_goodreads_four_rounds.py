"""User-run Goodreads-LP: fixed four sampled rounds, all ten seeds, full official test."""
import argparse
import copy
import datetime
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
OUT = ROOT / 'artifacts/goodreads_four_rounds'
PROFILE = ROOT / 'configs/goodreads_four_rounds_20261021.json'
RELEASE = ROOT / 'artifacts/delivery/goodreads_four_rounds_release.json'
LEDGER = ROOT / 'artifacts/workflow_state/goodreads_four_rounds_current.json'
PAUSE = ROOT / 'artifacts/workflow_state/goodreads_four_rounds_pause.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify():
    from came_repro.author_loader import fingerprint
    from run_21_goodreads_official import verify as verify_original
    p = verify_original()
    for file, digest in read(RELEASE)['input_sha256'].items():
        if fingerprint(file) != digest:
            raise ValueError('Frozen four-round input changed: ' + file)
    short = read(PROFILE)
    expected = dict(p['recipe'], max_rounds=4)
    if short['recipe'] != expected or short['seeds'] != list(range(10)) or short['test_score_admission_gate'] is not False:
        raise ValueError('Fixed four-round protocol changed')
    return p, short


def config_for(p, short, seed):
    from came_repro.author_loader import fingerprint
    from run_21_goodreads_official import config_for as original_config
    p = copy.deepcopy(p)
    p['recipe'] = short['recipe']
    config = original_config(p, seed)
    config['name'] = 'goodreads_four_round_head'
    config['evaluation'].update(four_round_release_sha256=fingerprint(RELEASE), shortened_budget=True)
    config['deviations'] = [s.replace('max30 rounds', 'max4 rounds (user time-budget adaptation)') for s in config['deviations']]
    config['deviations'].append('Four sampled rounds fixed before the seed0 diagnostic test; all ten seeds retained; not the original30-round upstream budget.')
    return config


def summarize(ledger):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from came_repro.atomic_io import atomic_write_json
    rows = sorted(ledger['results'].values(), key=lambda r: r['seed'])
    scores = [r['score_pct'] for r in rows]
    stats = dict(completed=len(rows), expected=10, optimization_rounds_per_seed=4,
        paper_pct=10.42, paper_protocol_equivalence_claimed=False, all_seeds_retained=True)
    if scores:
        stats.update(mean_pct=float(np.mean(scores)), gap_pp=float(np.mean(scores)-10.42),
            sample_sd_pp=float(np.std(scores, ddof=1)) if len(scores)>1 else None)
    atomic_write_json(OUT / 'results.json', dict(state=ledger['state'], statistics=stats, results=rows))
    if not rows:
        return
    with (OUT / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['seed'] for r in rows], scores, 'o-', label='Complete official test; four sampled rounds')
    ax.axhline(10.42, color='black', ls='--', label='Paper10.42')
    ax.set(xlabel='Downstream seed', ylabel='Official test MRR (%)', title='Goodreads-LP: %d/10 seeds' % len(rows))
    ax.legend(fontsize=8)
    fig.text(.02, .01, 'Full1710677 queries; fixed150 negatives;1024-wide frozen cache. Four-round budget is an adaptation.', fontsize=7)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(OUT / 'goodreads_four_rounds.png', dpi=180)
    plt.close(fig)


def main():
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import best_resume, run_child
    from came_repro.pause_control import TrainingPaused
    from run_21_goodreads_official import artifacts
    from remaining_delivery import audit_test
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--stage', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume-head', type=Path)
    parser.add_argument('--selection-lock', type=Path)
    args = parser.parse_args()
    if args.pause:
        atomic_write_json(PAUSE, dict(reason='user_paused_four_round_training'))
        return
    p, short = verify()
    if args.seed not in range(10):
        raise ValueError('Only fixed seeds0..9 allowed')
    if args.dry_run:
        print('Identity verified;4 sampled rounds per seed; full official splits; no GPU training started.', flush=True)
        return
    if args.stage:
        import torch
        from came_repro import goodreads_official as core
        from came_repro.experiment import Experiment
        from came_repro.protocol_repair import check_cuda
        check_cuda()
        torch.set_num_threads(4)
        if args.stage == 'test':
            core.test_saved(p, args.selection_lock, args.resume_head)
            return
        gpu, split, audit = core.load_data(p)
        e = Experiment(config_for(p, short, args.seed), audit, args.resume_head or Path(p['encoder_checkpoint']))
        try:
            s = core.train_rounds(gpu, split['train'], split['valid'], short['recipe'], args.seed, e, args.resume_head)
            s.update(encoder_sha256=p['encoder_sha256'], embedding_cache=p['embedding_cache'], shortened_budget=True)
            atomic_write_json(e.directory / 'summary.json', s)
            e.status('completed_four_round_validation')
        except BaseException as error:
            e.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error))
            raise
        finally:
            e.close()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    guard = acquire_gpu_lock(ROOT / '.runtime/goodreads_official.lock')
    logger = logging.getLogger('goodreads_four_rounds')
    logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler(), logging.FileHandler(OUT / 'pipeline.log', encoding='utf-8')]
    for handler in handlers:
        formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    ledger = read(LEDGER) if LEDGER.exists() else dict(state='ready', release_sha256=fingerprint(RELEASE), active_seconds=0., records={}, results={})
    sleep_guard = False
    def persist():
        atomic_write_json(LEDGER, ledger)
    try:
        if ledger['release_sha256'] != fingerprint(RELEASE):
            raise ValueError('Existing four-round experiment identity changed')
        quick = read(ROOT / 'artifacts/workflow_state/goodreads_quick_check_current.json')
        if quick['state'] != 'completed_short_diagnostic':
            raise ValueError('Wait for the real full official seed0 test and independent audit before training')
        for file, digest in quick['artifacts'].items():
            if fingerprint(file) != digest:
                raise ValueError('Real seed0 evidence changed')
        if '0' not in ledger['results']:
            result = audit_test('books', quick['result']['run_directory'])
            result.update(seed=0, run_directory=quick['result']['run_directory'], paper_pct=10.42,
                gap_pp=result['score_pct']-10.42, optimization_rounds=4, reused_completed_diagnostic=True)
            ledger['results']['0'] = result
            ledger['seed0_evidence'] = quick['artifacts']
            persist()
        if PAUSE.exists():
            history = OUT / 'pause_history'
            history.mkdir(exist_ok=True)
            atomic_write_json(history / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), read(PAUSE))
            PAUSE.unlink()  # Only this entry's flag; original30-round pause is retained.
        probe = acquire_gpu_lock()
        probe.close()
        from came_repro.goodreads_storage import reserve
        from came_repro.storage_budget import check_budget
        reserve(check_budget(additional_project_bytes=256*2**20, additional_cache_bytes=2**30))
        import ctypes
        sleep_guard = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        def execute(name, stage, seed, selection=None):
            verify()
            rec = ledger['records'].setdefault(name, dict(state='ready', attempts=[]))
            if rec['state'] == 'completed':
                for file, digest in rec['artifacts'].items():
                    if fingerprint(file) != digest:
                        raise ValueError('Completed evidence changed')
                return rec
            parent = best_resume(rec, 'link_head_last.pt' if stage=='train' else 'test_cursor.json')
            arguments = [str(Path(__file__)), '--stage', stage, '--seed', str(seed)]
            if parent:
                arguments += ['--resume-head', str(parent)]
            if selection:
                arguments += ['--selection-lock', str(selection)]
            ledger['state'] = 'running_' + name
            logger.info('GOODREADS_FOUR_ROUNDS seed=%d/9 stage=%s; completed=%d/10 parent=%s', seed, stage, len(ledger['results']), parent)
            code, run = run_child(arguments, PAUSE, ledger, LEDGER, rec, logger, float('inf'))
            if code == 75 or PAUSE.exists():
                raise TrainingPaused('Complete training/test boundary saved')
            if code or not run:
                raise RuntimeError('Four-round stage failed: %s exit=%s' % (name, code))
            rec.update(state='completed', run_directory=run, summary=read(Path(run)/'summary.json'), artifacts=artifacts(run))
            persist()
            return rec
        summarize(ledger)
        for seed in range(1, 10):
            key = str(seed)
            if key not in ledger['results']:
                trained = execute('train_s'+key, 'train', seed)
                run = Path(trained['run_directory'])
                head = Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
                selection = OUT / ('seed%d_selection_lock.json' % seed)
                value = dict(validation_run_directory=str(run), validation_summary_sha256=fingerprint(run/'summary.json'),
                    classifier_checkpoint=str(head), classifier_sha256=fingerprint(head), locked_before_test=True,
                    test_used_for_selection=False, selected_validation_mrr=trained['summary']['validation_mrr'])
                if selection.exists() and read(selection) != value:
                    raise ValueError('Frozen head selection cannot be replaced')
                atomic_write_json(selection, value)
                tested = execute('test_s'+key, 'test', seed, selection)
                result = audit_test('books', tested['run_directory'])
                result.update(seed=seed, run_directory=tested['run_directory'], paper_pct=10.42,
                    gap_pp=result['score_pct']-10.42, optimization_rounds=4, reused_completed_diagnostic=False)
                ledger['results'][key] = result
                persist()
            else:
                execute('test_s'+key, 'test', seed)  # Verify existing artifacts; never retest.
            summarize(ledger)
        ledger['state'] = 'completed_ten_seeds_four_rounds'
        persist()
        summarize(ledger)
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error, (TrainingPaused, KeyboardInterrupt)) else 'failed', error=str(error))
        persist()
        raise
    finally:
        if sleep_guard:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        guard.close()


if __name__ == '__main__':
    main()
