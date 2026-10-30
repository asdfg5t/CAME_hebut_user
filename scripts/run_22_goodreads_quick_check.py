"""Confirm the user's shortened diagnostic; never starts ten-seed training."""
import argparse
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)

OUT = ROOT / 'artifacts/goodreads_quick_check'
LEDGER = ROOT / 'artifacts/workflow_state/goodreads_quick_check_current.json'
PAUSE = ROOT / 'artifacts/workflow_state/goodreads_quick_check_pause.json'
SOURCE_RUN = ROOT / 'runs/20261021T102329Z_goodreads_of_893450_s0_fba8d7'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def promote():
    import numpy as np
    import torch
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.experiment import Experiment
    from came_repro.goodreads_official import draw_curves, metrics
    from run_21_goodreads_official import PROFILE

    directory = Path(read(SOURCE_RUN / 'checkpoints/location.json')['checkpoint_directory'])
    parent, selected = directory / 'link_head_last.pt', directory / 'link_head_best.pt'
    state = torch.load(str(parent), map_location='cpu', weights_only=False)
    best = torch.load(str(selected), map_location='cpu', weights_only=False)
    history = state['history_rows']
    if (len(history) != 4 or state['next_round'] != 5 or state['positive_cursor'] != 512000
            or state['selected_round'] != 2 or best['selected_round'] != 2):
        raise ValueError('The user-paused training boundary changed; review before promotion')
    if max(row['validation_mrr'] for row in history) != state['best_mrr'] or best['validation_mrr'] != state['best_mrr']:
        raise ValueError('Select only the highest COMPLETE validation, never a target test score')
    if set(best['head']) != set(state['best_state']):
        raise ValueError('Selected checkpoint keys changed')
    for key in best['head']:
        torch.testing.assert_close(best['head'][key], state['best_state'][key], rtol=0, atol=0)
    profile = read(PROFILE)
    split = torch.load(str(ROOT / 'data/raw/mm_graph/books-lp/lp-edge-split.pt'), map_location='cpu', weights_only=False, mmap=True)
    with np.load(SOURCE_RUN / 'validation_link_ranks.npz') as values:
        for key in ['source_node', 'target_node']:
            np.testing.assert_array_equal(values[key], split['valid'][key])
        if len(values['rank']) != 570464 or metrics(values['rank'])['mrr'] != state['best_mrr']:
            raise ValueError('Complete official validation ranks do not match the saved best head')
    config = copy.deepcopy(read(SOURCE_RUN / 'config.json'))
    config['name'] = 'goodreads_short_diagnostic_selection'
    config['evaluation']['shortened_diagnostic'] = dict(
        reason='User requested a shorter single-seed check before all30 rounds',
        completed_validations=4, selected_round=2, training_round5_preserved_unvalidated=True,
        formal_ten_seed_result=False, test_used_for_selection=False)
    summary = dict(dataset='books-lp', task='link', seed=0, validation_mrr=state['best_mrr'],
        selected_round=2, rounds_completed=4, test_evaluations=0, test=None,
        head_variant='pinned_sage_predictor_1024', recipe=profile['recipe'],
        official_train_pool=3437017, positive_budget_per_round=512000,
        full_official_epochs_claimed=False, paper_reproduction_complete=False,
        head_is_reconstruction=True, diagnostic_only=True, original_thirty_round_training_paused=True,
        original_training_run=str(SOURCE_RUN), resume_checkpoint=str(parent),
        resume_checkpoint_sha256=fingerprint(parent), selected_head_sha256=fingerprint(selected),
        encoder_sha256=profile['encoder_sha256'], embedding_cache=profile['embedding_cache'])
    e = Experiment(config, state['data_audit'], parent)
    try:
        e.logger.info('PROMOTED_SAVED_VALIDATION: no new training updates; source=%s; completed validations=4; best round=2; original round5 optimizer preserved', SOURCE_RUN)
        shutil.copyfile(SOURCE_RUN / 'validation_link_ranks.npz', e.directory / 'validation_link_ranks.npz')
        for row in history:
            e.metric(row)  # Explicit historical replay, not newly performed updates.
        draw_curves(history, 2, e)
        atomic_write_json(e.directory / 'summary.json', summary)
        e.status('completed_short_diagnostic_selection')
    finally:
        e.close()
    lock = dict(validation_run_directory=str(e.directory),
        validation_summary_sha256=fingerprint(e.directory / 'summary.json'),
        classifier_checkpoint=str(selected), classifier_sha256=fingerprint(selected),
        locked_before_test=True, test_used_for_selection=False,
        selected_validation_mrr=state['best_mrr'], diagnostic_only=True,
        parent_checkpoint_sha256=fingerprint(parent))
    path = OUT / 'selection_lock.json'
    atomic_write_json(path, lock)
    return path


def main():
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.pause_control import TrainingPaused
    from came_repro.quality_pipeline import best_resume, run_child
    from remaining_delivery import audit_test
    from run_21_goodreads_official import artifacts, verify, RELEASE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()
    if args.pause:
        atomic_write_json(PAUSE, dict(reason='user_paused_short_diagnostic'))
        return
    verify()
    OUT.mkdir(parents=True, exist_ok=True)
    identity = dict(helper_sha256=fingerprint(Path(__file__)), original_release_sha256=fingerprint(RELEASE))
    guard = acquire_gpu_lock(ROOT / '.runtime/goodreads_official.lock')
    logger = logging.getLogger('goodreads_quick_check')
    logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler(), logging.FileHandler(OUT / 'pipeline.log', encoding='utf-8')]
    for handler in handlers:
        formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    ledger = read(LEDGER) if LEDGER.exists() else dict(identity=identity, state='ready', active_seconds=0., records={})
    sleep_guard = False
    try:
        if ledger['identity'] != identity:
            raise ValueError('Short diagnostic source identity changed')
        if ledger['state'] == 'completed_short_diagnostic':
            for path, digest in ledger['artifacts'].items():
                if fingerprint(path) != digest:
                    raise ValueError('Completed diagnostic artifact changed')
            print(json.dumps(ledger['result'], ensure_ascii=False), flush=True)
            return
        if PAUSE.exists():
            raise TrainingPaused('Quick diagnostic pause retained; explicit resume audit required')
        probe = acquire_gpu_lock()
        probe.close()
        import ctypes
        sleep_guard = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        if 'selection_lock' not in ledger:
            ledger['selection_lock'] = str(promote())
            ledger['selection_lock_sha256'] = fingerprint(ledger['selection_lock'])
            atomic_write_json(LEDGER, ledger)
        if fingerprint(ledger['selection_lock']) != ledger['selection_lock_sha256']:
            raise ValueError('Frozen selection lock changed')
        record = ledger['records'].setdefault('test_s0', dict(state='ready', attempts=[]))
        parent = best_resume(record, 'test_cursor.json')
        arguments = [str(ROOT / 'scripts/run_21_goodreads_official.py'), '--stage', 'test',
            '--seed', '0', '--selection-lock', ledger['selection_lock']]
        if parent:
            arguments += ['--resume-head', str(parent)]
        ledger['state'] = 'running_complete_official_test'
        code, run = run_child(arguments, PAUSE, ledger, LEDGER, record, logger, float('inf'))
        if code == 75 or PAUSE.exists():
            raise TrainingPaused('Durable test query boundary saved')
        if code or not run:
            raise RuntimeError('Complete official diagnostic failed: exit=%s' % code)
        result = audit_test('books', run)
        result.update(seed=0, paper_pct=10.42, gap_pp=result['score_pct']-10.42,
            diagnostic_only=True, selected_round=2, completed_validation_rounds=4, run_directory=run)
        ledger.update(state='completed_short_diagnostic', result=result, artifacts=artifacts(run))
        record.update(state='completed', run_directory=run)
        atomic_write_json(LEDGER, ledger)
        atomic_write_json(OUT / 'result.json', result)
        logger.info('INDEPENDENTLY_VERIFIED %s', json.dumps(result, ensure_ascii=False))
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error, (TrainingPaused, KeyboardInterrupt)) else 'failed', error=str(error))
        atomic_write_json(LEDGER, ledger)
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
