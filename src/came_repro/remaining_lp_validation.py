"""Three declared LP single-seed pilots, serial CUDA, resumable into new runs."""
import datetime
import logging
from pathlib import Path

from .atomic_io import read_json_file, atomic_write_json
from .author_loader import ROOT, fingerprint, verify_originals
from .ide_workflow import acquire_gpu_lock
from .quality_pipeline import run_child, best_resume
from .storage_budget import check_budget


def collect_artifacts(run):
    run = Path(run)
    names = ['summary.json', 'manifest.json', 'config.json', 'status.json', 'logs/metrics.csv',
        'logs/metrics.jsonl', 'cuda_row_cache_equivalence.json', 'test_link_ranks.npz',
        'reverse_overlap_queries.npz', 'checkpoints/location.json']
    location = read_json_file(run/'checkpoints/location.json')
    checkpoint = Path(location['checkpoint_directory'])
    files = [run/n for n in names]+list((run/'figures').glob('*.png'))
    files += [checkpoint/'link_head_best.pt', checkpoint/'link_head_last.pt']
    return {str(p):fingerprint(p) for p in files}


def verify_record(record):
    for path, digest in record['artifact_sha256'].items():
        if fingerprint(path) != digest:
            raise ValueError('Completed LP artifact changed: '+path)
    import numpy as np
    run = Path(record['run_directory'])
    saved = np.load(run/'test_link_ranks.npz')
    rank = saved['rank'].astype(np.float64)
    if not np.isfinite(rank).all() or not (rank >= 1).all():
        raise ValueError('Invalid official LP ranks')
    result = record['summary']['test']
    if len(rank) != result['queries'] or abs(float(np.mean(1/rank))-result['official']['mrr']) > 1e-7:
        raise ValueError('Stored ranks disagree with official summary')
    if record['summary']['test_evaluations'] != 1:
        raise ValueError('LP test should be evaluated once after validation selection')


def main():
    verify_originals()
    kg = read_json_file(ROOT/'artifacts/workflow_state/kg_affine_screen_current.json')
    if kg['state'] != 'completed_two_kg_diagnostics':
        raise RuntimeError('Complete current KG diagnostics before dispatching LP')
    profile_path = ROOT/'configs/remaining_lp_single_seed.json'
    profile = read_json_file(profile_path)
    if (profile['seed'] != 0 or profile['training']['epochs'] != 100 or profile['training']['patience'] != 10
            or profile['training']['lr'] != .001 or not profile['evaluation']['all_official_candidates']):
        raise ValueError('Audit changed declared LP protocol before running')
    model = Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(model) != kg['identity']['encoder_sha256']:
        raise ValueError('Frozen encoder identity changed')
    sources = {str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    identity = {'encoder_sha256':fingerprint(model), 'sources':sources, 'profile':profile,
        'kg_confirmation_ledger_sha256':fingerprint(ROOT/'artifacts/workflow_state/kg_affine_screen_current.json'),
        'test_used_for_selection':False}
    path = ROOT/'artifacts/workflow_state/remaining_lp_validation_current.json'
    ledger = read_json_file(path) if path.is_file() else {'identity':identity, 'state':'ready', 'active_seconds':0., 'records':{}}
    if ledger['identity'] != identity:
        raise ValueError('LP identity changed; audit before resuming')
    for record in ledger['records'].values():
        if record['state'] == 'completed': verify_record(record)
    directory = ROOT/'artifacts/remaining_main_pilots/lp_single_seed'
    directory.mkdir(parents=True, exist_ok=True)
    lock = acquire_gpu_lock(ROOT/'.runtime/remaining_lp_validation.lock')
    logger = logging.getLogger('remaining_lp_validation');logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')):
        handler.setFormatter(formatter);logger.addHandler(handler)
    pause = ROOT/'artifacts/workflow_state/remaining_lp_validation_pause.json'
    try:
        ledger['budget_at_dispatch'] = check_budget(); atomic_write_json(path, ledger)
        for task in profile['tasks']:
            dataset = task['dataset']
            for source, digest in sources.items():
                if fingerprint(ROOT/source) != digest: raise ValueError('Active LP source changed')
            record = ledger['records'].setdefault(dataset, {'state':'ready', 'attempts':[]})
            if record['state'] == 'completed': continue
            args = ['-m','came_repro.remaining_lp_pilot', '--dataset',dataset, '--checkpoint',str(model),
                    '--seed','0','--epochs','100','--patience','10','--inference-batch-size','32']
            resume = best_resume(record, 'link_head_last.pt')
            if resume: args += ['--resume-head',str(resume)]
            ledger['state'] = 'running'
            code, run = run_child(args,pause,ledger,path,record,logger,float('inf'))
            if code or not run:
                record['state'] = 'paused' if code == 75 else 'failed'
                ledger['state'] = record['state'];atomic_write_json(path,ledger)
                raise RuntimeError('LP child '+dataset+' exited '+str(code))
            record.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_artifacts(run))
            verify_record(record)
            record['paper_mrr_pct'] = task['paper_mrr_pct']
            record['gap_pp'] = 100*record['summary']['test']['official']['mrr']-task['paper_mrr_pct']
            record['within_two_pp'] = abs(record['gap_pp']) <= 2
            atomic_write_json(path,ledger)
            logger.info('FINAL dataset=%s official_MRR_pct=%.8f gap_pp=%.8f',dataset,100*record['summary']['test']['official']['mrr'],record['gap_pp'])
        ledger['state'] = 'completed_three_lp_diagnostics'
        ledger['remaining_unstarted'] = ['ogbn-products']
        atomic_write_json(path,ledger)
    except BaseException as error:
        if ledger['state'] != 'paused': ledger['state'] = 'failed'
        ledger['error'] = str(error);atomic_write_json(path,ledger);raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
