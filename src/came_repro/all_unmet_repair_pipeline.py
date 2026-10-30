"""Serial bounded protocol repair. Passing results and historical ledgers stay unchanged."""
import copy
import datetime
import logging
from pathlib import Path

from .atomic_io import read_json_file, atomic_write_json
from .author_loader import ROOT, fingerprint, verify_originals
from .ide_workflow import acquire_gpu_lock
from .quality_pipeline import run_child, best_resume
from .remaining_kg_validation import verify_records, artifact_hashes
from .storage_budget import check_budget
from .protocol_repair import PROFILE, candidate_cache


def validation_improves(candidate, reference, metric, minimum_gain_pp):
    """Never consult test scores, paper targets, or seed alternatives."""
    import math
    a, b = candidate['summary'][metric], reference['summary'][metric]
    if not all(math.isfinite(v) for v in (a, b, minimum_gain_pp)) or minimum_gain_pp < 0:
        raise ValueError('Finite validation gate required')
    return 100*(a-b) >= minimum_gain_pp


def collect_protocol_artifacts(run):
    run = Path(run); hashes = artifact_hashes(run)
    for name in ['inference_state.json', 'candidate_feature_audit.json', 'test_link_ranks.npz', 'validation_link_ranks.npz', 'reverse_overlap_queries.npz', 'train_affine_statistics.npz', 'checkpoints/location.json']:
        path = run/name
        if path.exists(): hashes[str(path)] = fingerprint(path)
    location = run/'checkpoints/location.json'
    if location.exists():
        directory = Path(read_json_file(location)['checkpoint_directory'])
        for name in ['link_head_best.pt', 'link_head_last.pt', 'probe_best.pt', 'probe_last.pt', 'probe_best_raw_linear.pt']:
            path = directory/name
            if path.exists(): hashes[str(path)] = fingerprint(path)
    return hashes


def main():
    verify_originals(); profile = read_json_file(PROFILE)
    acceptance_path = ROOT/'artifacts/audit/protocol_repair_acceptance_20261014.json'
    acceptance = read_json_file(acceptance_path)
    if not acceptance['passed'] or acceptance['profile_sha256'] != fingerprint(PROFILE):
        raise ValueError('Frozen CUDA/numerical acceptance required before candidates')
    for filename, digest in acceptance['sources'].items():
        if fingerprint(ROOT/filename) != digest: raise ValueError('Accepted source changed')
    model = Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(model) != profile['encoder_sha256']: raise ValueError('Protected encoder changed')
    old_lp = read_json_file(ROOT/'artifacts/workflow_state/remaining_lp_validation_current.json')
    repair = read_json_file(ROOT/'artifacts/workflow_state/validation_repair_current.json')
    if old_lp['state'] != 'completed_three_lp_diagnostics' or repair['state'] != 'completed_bounded_validation_repairs':
        raise ValueError('Finish prior diagnostics before protocol repair')
    verify_records(old_lp); verify_records(repair)
    sources = {str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    identity = {'sources':sources, 'profile':profile, 'encoder_sha256':fingerprint(model),
        'reference_lp_sha256':fingerprint(ROOT/'artifacts/workflow_state/remaining_lp_validation_current.json'),
        'reference_repair_sha256':fingerprint(ROOT/'artifacts/workflow_state/validation_repair_current.json'),
        'acceptance_sha256':fingerprint(acceptance_path)}
    path = ROOT/'artifacts/workflow_state/all_unmet_repair_current.json'
    ledger = read_json_file(path) if path.exists() else {'identity':identity,'state':'ready','records':{},'selections':{},'active_seconds':0.}
    if ledger['identity'] != identity: raise ValueError('Protocol repair identity changed; audit before resume')
    verify_records(ledger)
    directory = ROOT/'artifacts/remaining_main_pilots/protocol_repair';directory.mkdir(parents=True,exist_ok=True)
    pause = ROOT/'artifacts/workflow_state/all_unmet_repair_pause.json'
    lock = acquire_gpu_lock(ROOT/'.runtime/all_unmet_repair.lock')
    logger = logging.getLogger('all_unmet_repair');logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in [logging.StreamHandler(),logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')]:
        handler.setFormatter(fmt);logger.addHandler(handler)
    storage = read_json_file(ROOT/'configs/storage.json')

    def execute(jid, arguments, resume_file=None):
        for source, digest in sources.items():
            if fingerprint(ROOT/source) != digest: raise ValueError('Activity source changed')
        if fingerprint(PROFILE) != acceptance['profile_sha256']: raise ValueError('Activity profile changed')
        record = ledger['records'].setdefault(jid, {'state':'ready','attempts':[]})
        if record['state'] == 'completed': return record
        resume = best_resume(record,resume_file) if resume_file else None
        if resume: arguments += ['--resume-head',str(resume)]
        ledger['state'] = 'running'
        code, run = run_child(arguments,pause,ledger,path,record,logger,float('inf'))
        if code or not run:
            record['state'] = ledger['state'] = 'paused' if code == 75 else 'failed'
            atomic_write_json(path,ledger);raise RuntimeError('Protocol child '+jid+' exited '+str(code))
        record.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_protocol_artifacts(run))
        atomic_write_json(path,ledger);return record

    def freeze_confirmation(dataset, candidate, metric):
        run = Path(candidate['run_directory']); summary = candidate['summary']; cache = candidate_cache(dataset,fingerprint(model),storage)
        checkpoint_directory = Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])
        best = checkpoint_directory/('probe_best_raw_linear.pt' if dataset == 'WN18RR' else 'link_head_best.pt')
        selected = {'dataset':dataset,'encoder_sha256':fingerprint(model),'classifier_checkpoint':str(best),
            'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),'validation_summary_sha256':fingerprint(run/'summary.json'),
            'embedding_cache':str(cache),'inference_state_sha256':fingerprint(cache/'inference_state.json'),
            'locked_before_test':True,'test_used_for_selection':False,'selected_validation_'+('accuracy' if dataset == 'WN18RR' else 'mrr'):summary[metric]}
        if dataset == 'WN18RR': selected.update(graph_policy='train_reverse',feature_transform='train_pair_affine')
        else: selected.update(head_variant=summary['head_variant'],graph_policy='upstream_train_bidirected_unique')
        selected_path = directory/(dataset+'_selection_lock.json')
        if selected_path.exists() and read_json_file(selected_path) != selected: raise ValueError('Selection lock changed')
        atomic_write_json(selected_path,selected);ledger['selections'][dataset]=selected;atomic_write_json(path,ledger)
        arguments = ['-m','came_repro.kg_selected_confirmation','--selection-lock',str(selected_path)] if dataset == 'WN18RR' else ['-m','came_repro.protocol_repair','--stage','confirm_lp','--selection-lock',str(selected_path)]
        return execute('confirm_'+dataset,arguments)

    try:
        ledger['budget_at_dispatch'] = check_budget(additional_cache_bytes=profile['resources']['additional_cache_reservation_bytes']);atomic_write_json(path,ledger)
        for dataset in ['sports-copurchase','cloth-copurchase','WN18RR']:
            reference = repair['records']['confirm_'+dataset]
            control = None
            if dataset == 'cloth-copurchase':
                control = execute('matched_directed_control_'+dataset,
                    ['-m','came_repro.protocol_repair','--stage','validate_lp','--dataset',dataset,'--checkpoint',str(model),'--directed-control'], 'link_head_last.pt')
            execute('encode_'+dataset,['-m','came_repro.protocol_repair','--stage','encode','--dataset',dataset,'--checkpoint',str(model)])
            if dataset == 'WN18RR':
                arguments = ['-m','came_repro.kg_affine_probe','--dataset',dataset,'--checkpoint',str(model),
                    '--embedding-cache',str(candidate_cache(dataset,fingerprint(model),storage)),'--graph-policy','train_reverse']
                metric, resume_file = 'validation_accuracy','probe_last.pt'
            else:
                arguments = ['-m','came_repro.protocol_repair','--stage','validate_lp','--dataset',dataset,'--checkpoint',str(model)]
                metric, resume_file = 'validation_mrr','link_head_last.pt'
            candidate = execute('validate_'+dataset,arguments,resume_file)
            if candidate['summary']['test_evaluations'] != 0: raise ValueError('Candidate must be validation-only')
            passes_control = control is None or validation_improves(candidate,control,metric,profile['minimum_validation_gain_pp'])
            if validation_improves(candidate,reference,metric,profile['minimum_validation_gain_pp']) and passes_control:
                confirmed = freeze_confirmation(dataset,candidate,metric)
                value = confirmed['summary']['test_accuracy'] if dataset == 'WN18RR' else confirmed['summary']['test']['official']['mrr']
                paper = {'sports-copurchase':37.68,'cloth-copurchase':28.88,'WN18RR':81.97}[dataset]
                confirmed.update(paper_pct=paper,gap_pp=100*value-paper,within_two_pp=abs(100*value-paper)<=2)
            else:
                ledger['records']['confirm_'+dataset] = copy.deepcopy(reference)
                ledger['selections'][dataset] = {'reuse_reference':True,'reason':'candidate failed predeclared validation gain gate','candidate_validation':candidate['summary'][metric],'reference_validation':reference['summary'][metric]}
            atomic_write_json(path,ledger)
        ledger.update(state='completed_bounded_graph_text_screen',remaining_protocol_review=['books-lp'],remaining_unmeasured=['ogbn-products'],all_unmet_repaired=False)
        atomic_write_json(path,ledger)
    except BaseException as error:
        if ledger['state'] != 'paused':ledger['state']='failed'
        ledger['error']=str(error);atomic_write_json(path,ledger);raise
    finally:lock.close()


if __name__=='__main__':main()
