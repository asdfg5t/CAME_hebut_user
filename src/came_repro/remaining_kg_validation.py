"""Recover KG validation, compare declared affine candidates, then test saved heads."""
import datetime
import logging
from pathlib import Path

from .atomic_io import read_json_file, atomic_write_json
from .author_loader import ROOT, fingerprint, verify_originals
from .ide_workflow import acquire_gpu_lock
from .quality_pipeline import run_child, best_resume
from .quality_probe import owned_cache_path


def artifact_hashes(run):
    run = Path(run)
    names = ['summary.json', 'manifest.json', 'config.json', 'status.json', 'logs/metrics.jsonl', 'logs/metrics.csv']
    files = [run/name for name in names] + list(run.glob('*predictions.npz')) + list((run/'figures').glob('*.png'))
    return {str(path):fingerprint(path) for path in files if path.is_file()}


def select_candidate(candidates, reference, minimum_gain_pp=.5):
    """Predeclared tie ordering; test scores are never read here."""
    best = max(candidates, key=lambda row:row['summary']['validation_accuracy'])
    gain = 100*(best['summary']['validation_accuracy']-reference['summary']['validation_accuracy'])
    return best if gain >= minimum_gain_pp else reference


def verify_records(ledger):
    for record in ledger.get('records', {}).values():
        if record.get('state') == 'completed':
            for path, digest in record.get('artifact_sha256', {}).items():
                if fingerprint(path) != digest:
                    raise ValueError('Completed diagnostic artifact changed: '+path)


def main():
    verify_originals()
    graph_path = ROOT/'artifacts/workflow_state/kg_graph_screen_current.json'
    graph = read_json_file(graph_path)
    for path, digest in graph['identity']['sources'].items():
        if fingerprint(ROOT/path) != digest:
            raise ValueError('Previous KG protocol source changed: '+path)
    verify_records(graph)
    model = Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(model) != graph['identity']['encoder_sha256']:
        raise ValueError('Frozen encoder changed')
    storage = read_json_file(ROOT/'configs/storage.json')
    directory = ROOT/'artifacts/remaining_main_pilots/kg_affine_screen'
    directory.mkdir(parents=True, exist_ok=True)
    path = ROOT/'artifacts/workflow_state/kg_affine_screen_current.json'
    sources = {str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    identity = {'sources':sources, 'encoder_sha256':fingerprint(model), 'seed':0,
        'candidates':['original_train_pair_affine', 'train_reverse_train_pair_affine'], 'minimum_validation_gain_pp':.5,
        'test_used_for_selection':False, 'epochs':2000, 'patience':100, 'lr':.05}
    ledger = read_json_file(path) if path.is_file() else {'identity':identity, 'state':'ready', 'active_seconds':0., 'records':{}, 'selections':{}}
    if ledger['identity'] != identity:
        raise ValueError('Diagnostic protocol changed; audit before resuming')
    verify_records(ledger)
    lock = acquire_gpu_lock(ROOT/'.runtime/remaining_kg_validation.lock')
    logger = logging.getLogger('remaining_kg_validation'); logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'), encoding='utf-8')):
        handler.setFormatter(formatter); logger.addHandler(handler)
    pause = ROOT/'artifacts/workflow_state/remaining_kg_validation_pause.json'

    def execute(jid, arguments, owner=ledger, owner_path=path):
        for filename, digest in sources.items():
            if fingerprint(ROOT/filename) != digest:
                raise ValueError('Active diagnostic source changed')
        record = owner.setdefault('records', {}).setdefault(jid, {'state':'ready', 'attempts':[]})
        if record['state'] == 'completed':
            return record
        resume = best_resume(record, 'probe_last.pt')
        if resume and '-m' in arguments and arguments[1] != 'came_repro.kg_selected_confirmation':
            arguments += ['--resume-head', str(resume)]
        owner['state'] = 'running'
        code, run = run_child(arguments, pause, owner, owner_path, record, logger, float('inf'))
        if code or not run:
            record['state'] = 'paused' if code == 75 else 'failed'
            owner['state'] = record['state']; atomic_write_json(owner_path, owner)
            raise RuntimeError('Diagnostic child '+jid+' exited '+str(code))
        record.update(state='completed', run_directory=run, summary=read_json_file(Path(run)/'summary.json'), artifact_sha256=artifact_hashes(run))
        atomic_write_json(owner_path, owner)
        return record

    try:
        # All previous source hashes match. The dead child resumes its last epoch checkpoint into a new run.
        for policy in ('original', 'train_reverse'):
            jid = 'validate_FB15K237_'+policy
            execute(jid, ['-m','came_repro.kg_graph_probe','--dataset','FB15K237','--checkpoint',str(model),'--graph-policy',policy,'--seed','0','--inference-batch-size','32'], graph, graph_path)
        graph['state'] = 'completed_validation_grid'
        atomic_write_json(graph_path, graph)
        for dataset in ('WN18RR', 'FB15K237'):
            if dataset == 'WN18RR':
                # Directed first pilot is a reference, but its test result is not a selection input.
                run = Path(graph['wn_original_reference']['run_directory'])
                original = {'run_directory':str(run), 'summary':read_json_file(run/'summary.json')}
            else:
                original = graph['records']['validate_'+dataset+'_original']
            reverse = graph['records']['validate_'+dataset+'_train_reverse']
            reference = select_candidate([reverse], original)
            candidates = []
            for policy in ('original', 'train_reverse'):
                cache = owned_cache_path(storage, fingerprint(model), dataset, policy)
                if dataset == 'WN18RR' and policy == 'original':
                    cache = Path(storage['embedding_cache_root'])/fingerprint(model)[:20]/dataset
                candidates.append(execute('validate_'+dataset+'_'+policy+'_affine', ['-m','came_repro.kg_affine_probe', '--dataset',dataset,'--checkpoint',str(model),'--embedding-cache',str(cache),'--graph-policy',policy,'--seed','0']))
            selected = select_candidate(candidates, reference)
            row = selected['summary']; validation_run = Path(selected['run_directory'])
            transform = row.get('feature_transform', 'raw')
            if dataset == 'WN18RR' and transform == 'raw' and row.get('graph_policy') == 'train_reverse':
                prior = graph['records']['confirm_WN18RR_train_reverse']
                if prior['summary']['validation_accuracy'] != row['validation_accuracy']:
                    raise ValueError('Previously confirmed WN validation changed')
                ledger['records']['confirm_WN18RR'] = prior
                ledger['selections'][dataset] = {'reused_previously_tested_reference':True, 'run_directory':prior['run_directory']}
                atomic_write_json(path, ledger); continue
            # The first directed WN pilot already had its diagnostic test; reuse it if selected.
            if row['test_evaluations']:
                ledger['selections'][dataset] = {'reused_previously_tested_reference':True, 'run_directory':str(validation_run)}
                atomic_write_json(path, ledger); continue
            checkpoint_dir = Path(read_json_file(validation_run/'checkpoints/location.json')['checkpoint_directory'])
            best = checkpoint_dir/('probe_best_raw_linear.pt' if transform == 'train_pair_affine' else 'probe_best.pt')
            policy = row.get('graph_policy', 'original')
            cache = owned_cache_path(storage, fingerprint(model), dataset, policy)
            selection = {'dataset':dataset, 'encoder_sha256':fingerprint(model), 'graph_policy':policy,
                'feature_transform':transform, 'classifier_checkpoint':str(best), 'classifier_sha256':fingerprint(best),
                'validation_run_directory':str(validation_run), 'validation_summary_sha256':fingerprint(validation_run/'summary.json'),
                'selected_validation_accuracy':row['validation_accuracy'], 'reference_validation_accuracy':reference['summary']['validation_accuracy'],
                'candidate_validation_accuracies':[c['summary']['validation_accuracy'] for c in candidates],
                'minimum_gain_pp':.5, 'locked_before_test':True, 'test_used_for_selection':False, 'embedding_cache':str(cache)}
            if dataset == 'WN18RR' and policy == 'original':
                selection['embedding_cache'] = str(Path(storage['embedding_cache_root'])/fingerprint(model)[:20]/dataset)
            selection_path = directory/(dataset+'_selection_lock.json')
            if dataset in ledger['selections'] and ledger['selections'][dataset] != selection:
                raise ValueError('Cannot replace a frozen selection')
            ledger['selections'][dataset] = selection
            atomic_write_json(selection_path, selection); atomic_write_json(path, ledger)
            final = execute('confirm_'+dataset, ['-m','came_repro.kg_selected_confirmation','--selection-lock',str(selection_path)])
            logger.info('FINAL dataset=%s test=%.8f validation=%.8f',dataset,final['summary']['test_accuracy'],row['validation_accuracy'])
        ledger['state'] = 'completed_two_kg_diagnostics'
        ledger['remaining_unstarted'] = ['sports-copurchase','cloth-copurchase','books-lp','ogbn-products']
        atomic_write_json(path, ledger)
    except BaseException as error:
        if ledger['state'] != 'paused': ledger['state'] = 'failed'
        ledger['error'] = str(error); atomic_write_json(path, ledger)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
