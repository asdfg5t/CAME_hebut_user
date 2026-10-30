"""Run one fresh joint-coverage candidate and two seed0 diagnostics serially.

This is a finite diagnostic, not a formal-repeat entry or recurring automation.
"""
import logging
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child, best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from priority_refinement_backend import reference_cache
    from run_priority_cloth_wn_validation import validation_passes
    from audit_two_lp_confirmation import audit_confirmation
    from representation_coverage_backend import PLAN, CONFIG, DIRECTORY
    plan = read_json_file(PLAN); verify_originals()
    if fingerprint(CONFIG) != plan['configuration_sha256']: raise ValueError('Declared configuration changed')
    sources = {str(p.resolve()): fingerprint(p) for p in (ROOT / 'src/came_repro').glob('*.py')}
    for name in ('scripts/run_representation_coverage_validation.py', 'scripts/representation_coverage_backend.py',
                 'scripts/run_lp_hard_negative_candidate.py', 'scripts/priority_refinement_backend.py',
                 'scripts/run_priority_cloth_wn_validation.py', 'scripts/audit_two_lp_confirmation.py', 'configs/storage.json'):
        p = ROOT / name; sources[str(p.resolve())] = fingerprint(p)
    identity = {'sources': sources, 'plan_sha256': fingerprint(PLAN), 'config_sha256': fingerprint(CONFIG)}
    path = ROOT / 'artifacts/workflow_state/representation_coverage_current.json'
    pause = ROOT / 'artifacts/workflow_state/representation_coverage_pause.json'
    ledger = read_json_file(path) if path.exists() else {'identity': identity, 'state': 'prepared', 'records': {},
        'results': {}, 'rejections': {}, 'active_seconds': 0., 'formal_repeats_started': 0}
    if identity != ledger['identity']: raise ValueError('Frozen diagnostic identity changed; audit before restoring')
    guard = acquire_gpu_lock(ROOT / '.runtime/representation_coverage.lock')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('coverage'); logger.setLevel(logging.INFO)
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for handler in (logging.StreamHandler(), logging.FileHandler(DIRECTORY / ('pipeline_' + stamp + '.log'), encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s')); logger.addHandler(handler)
    backend = ROOT / 'scripts/representation_coverage_backend.py'
    def verify():
        if pause.exists(): raise RuntimeError('Diagnostic pause retained; explicit audited resume required')
        verify_originals()
        if fingerprint(PLAN) != identity['plan_sha256'] or fingerprint(CONFIG) != identity['config_sha256']:
            raise ValueError('Frozen plan changed')
        for name, digest in sources.items():
            if fingerprint(name) != digest: raise ValueError('Frozen source changed: ' + name)
        for record in plan['references'].values():
            if fingerprint(Path(record['run']) / 'summary.json') != record['summary_sha256']:
                raise ValueError('Fixed prior reference changed')
    def execute(job, arguments, resume_filename=None, resume_flag='--resume-head', acceptance_file=None):
        verify()
        record = ledger['records'].setdefault(job, {'state': 'ready', 'attempts': []})
        if record['state'] == 'completed':
            for name, digest in record['artifact_sha256'].items():
                if fingerprint(name) != digest: raise ValueError('Completed artifact changed')
            return record
        parent = best_resume(record, resume_filename) if resume_filename else None
        if parent: arguments += [resume_flag, str(parent)]
        if job.startswith('confirm_'):
            for attempt in record['attempts']:
                run = attempt.get('run_directory')
                if run and any((Path(run) / n).exists() for n in ('test_predictions.npz', 'test_link_ranks.npz')):
                    raise ValueError('Interrupted test output exists; independently audit instead of retesting')
        ledger['state'] = 'running_' + job
        code, run = run_child(arguments, pause, ledger, path, record, logger, float('inf'))
        if code:
            ledger['state'] = 'paused' if code == 75 else 'failed'
            raise RuntimeError('Finite coverage diagnostic stopped: ' + job)
        if acceptance_file:
            accepted = read_json_file(acceptance_file)
            if not accepted['passed'] or accepted['backend_sha256'] != fingerprint(backend):
                raise ValueError('Actual CUDA acceptance invalid')
            record.update(state='completed', artifact_sha256={str(acceptance_file): fingerprint(acceptance_file)}, summary=accepted)
        else:
            if not run: raise ValueError('Completed experiment has no unique run directory')
            hashes = collect_protocol_artifacts(run)
            location = Path(run) / 'checkpoints/location.json'
            if location.exists():
                directory = Path(read_json_file(location)['checkpoint_directory'])
                for name in ('last.pt', 'encoder_final.pt', 'link_head_last.pt', 'probe_last.pt'):
                    p = directory / name
                    if p.exists(): hashes[str(p)] = fingerprint(p)
            record.update(state='completed', run_directory=run, summary=read_json_file(Path(run) / 'summary.json'), artifact_sha256=hashes)
        atomic_write_json(path, ledger)
        return record
    try:
        verify(); probe = acquire_gpu_lock(); probe.close()
        execute('CUDA_capacity', [str(backend), '--stage', 'calibration'], acceptance_file=DIRECTORY / 'CUDA_capacity_acceptance.json')
        execute('joint_resume_acceptance', [str(backend), '--stage', 'resume_acceptance'], acceptance_file=DIRECTORY / 'joint_resume_acceptance.json')
        trained = execute('fresh_pretraining', [str(backend), '--stage', 'pretrain'], 'last.pt', '--resume')
        if not trained['summary'].get('complete_declared_subset_pretraining') or trained['summary']['completed_subset_rounds'] != 8 or trained['summary']['step'] != plan['expected_updates']:
            raise ValueError('Exactly eight complete declared subset rounds required before downstream evaluation')
        run = Path(trained['run_directory'])
        encoder = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / 'encoder_final.pt'
        saved = torch.load(str(encoder), map_location='cpu', weights_only=False)
        if saved['schema'] != 'came_encoder_v1' or saved['step'] != 20480 or saved['completed_subset_rounds'] != 8:
            raise ValueError('Complete matching fresh encoder required')
        del saved
        for task in ('cloth', 'wn'):
            execute('encode_' + task, [str(backend), '--stage', 'encode_' + task, '--checkpoint', str(encoder)])
            selected = execute('validate_' + task, [str(backend), '--stage', 'probe_' + task, '--checkpoint', str(encoder)],
                               'link_head_last.pt' if task == 'cloth' else 'probe_last.pt')
            key = 'validation_mrr' if task == 'cloth' else 'validation_accuracy'
            summary = selected['summary']; gain = 100 * (summary[key] - plan['reference_validation'][task])
            if not validation_passes(summary, plan['reference_validation'][task], key, plan['minimum_validation_gain_pp']):
                ledger['rejections'][task] = {'validation_gain_pp': gain, 'new_test_evaluations': 0, 'previous_confirmed_result_retained': True}
                atomic_write_json(path, ledger); continue
            run = Path(selected['run_directory']); cache = reference_cache(summary, read_json_file(run / 'config.json'))
            classifier = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / ('link_head_best.pt' if task == 'cloth' else 'probe_best_raw_linear.pt')
            lock = {'dataset': summary['dataset'], 'encoder_sha256': fingerprint(encoder), 'classifier_checkpoint': str(classifier),
                    'classifier_sha256': fingerprint(classifier), 'validation_run_directory': str(run), 'validation_summary_sha256': fingerprint(run / 'summary.json'),
                    'embedding_cache': str(cache), 'inference_state_sha256': fingerprint(cache / 'inference_state.json'),
                    'locked_before_test': True, 'test_used_for_selection': False, 'graph_policy': summary['graph_policy']}
            if task == 'cloth': lock.update(head_variant=summary['head_variant'], selected_validation_mrr=summary[key])
            else: lock.update(feature_transform='train_pair_affine', selected_validation_accuracy=summary[key])
            lock_path = DIRECTORY / (task + '_selection_lock.json')
            if lock_path.exists() and read_json_file(lock_path) != lock: raise ValueError('Frozen selection lock changed')
            atomic_write_json(lock_path, lock)
            args = ['-m', 'came_repro.protocol_repair', '--stage', 'confirm_lp', '--selection-lock', str(lock_path)] if task == 'cloth' else ['-m', 'came_repro.kg_selected_confirmation', '--selection-lock', str(lock_path)]
            confirmed = execute('confirm_' + task, args)
            if task == 'cloth': result = audit_confirmation(Path(confirmed['run_directory']), lock_path)
            else:
                from came_repro.kg_graph_probe import kg_input
                matrix = np.load(cache / 'fused_embeddings.npy', mmap_mode='r'); official = kg_input('WN18RR', matrix, 'test')
                with np.load(Path(confirmed['run_directory']) / 'test_predictions.npz') as values:
                    np.testing.assert_array_equal(values['pairs'], official.pairs); np.testing.assert_array_equal(values['labels'], official.labels)
                    score = float(np.mean(values['prediction'] == official.labels))
                assert len(official) == 3134 and score == confirmed['summary']['test_accuracy']
                result = {'full_official_pairs_labels_verified': True, 'test_accuracy_pct': score*100,
                          'paper_accuracy_pct': 81.97, 'gap_pp': score*100-81.97,
                          'within_two_pp': abs(score*100-81.97) <= 2, 'test_evaluations': 1}
            result.update(validation_gain_pp=gain, manual_log_checkpoint_figure_review_pending=True)
            ledger['results'][task] = result; atomic_write_json(path, ledger)
        ledger.update(state='completed_finite_coverage_diagnostic_pending_independent_review', formal_repeats_started=0,
                      all_targets_met=all(ledger['results'].get(t, {}).get('within_two_pp', False) for t in ('wn', 'cloth')))
        atomic_write_json(path, ledger)
    except BaseException as error:
        ledger.update(state='paused' if pause.exists() or ledger.get('state') == 'paused' else 'failed', error=str(error))
        atomic_write_json(path, ledger); raise
    finally: guard.close()


if __name__ == '__main__': main()
