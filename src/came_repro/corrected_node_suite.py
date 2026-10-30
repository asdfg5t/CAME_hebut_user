"""Repeat the validation-locked four-task protocol, preserving historical runs."""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

from .author_loader import ROOT, fingerprint, verify_originals
from .atomic_io import atomic_write_json, read_json_file
from .ide_workflow import acquire_gpu_lock
from .quality_pipeline import best_resume, reference_assets, run_child


def build_jobs(profile, single_seed=False, stability_check=False):
    if single_seed and stability_check:
        raise ValueError('Choose one explicit evaluation scope')
    if (profile['datasets'] != ['ele-fashion', 'wiki-cs', 'ogbn-arxiv', 'books-nc']
            or profile['evaluation_seeds'] != list(range(10))
            or profile['wiki_seeds'] != list(range(20))
            or profile['wiki_splits'] != list(range(20))
            or profile['graph_policies'] != dict(zip(profile['datasets'], ['original']+['bidirected']*3))
            or profile['probe_epochs'] != 2000 or profile['probe_patience'] != 100
            or profile['extra_pretraining_updates'] != 0 or profile['cumulative_time_limit'] is not None):
        raise ValueError('Keep the frozen four-task protocol and explicit paper repeat grid')
    jobs = []
    for dataset in profile['datasets']:
        seeds = profile['wiki_seeds'] if dataset == 'wiki-cs' else profile['evaluation_seeds']
        splits = profile['wiki_splits'] if dataset == 'wiki-cs' else [0]
        if single_seed and dataset != 'ele-fashion':
            seeds, splits = [0], [0]
        elif stability_check and dataset != 'ele-fashion':
            seeds, splits = [0, 1, 2], [0]
        for split in splits:
            for seed in seeds:
                jobs.append({'id': f'nc_{dataset}_{split}_{seed}', 'dataset': dataset,
                             'seed': seed, 'split_id': split, 'policy': profile['graph_policies'][dataset]})
    return jobs


def verify_record(record, job, root=ROOT):
    """Verify artifacts AND complete official test membership, without choosing by score."""
    import numpy as np
    if record['state'] != 'completed':
        raise ValueError('Cannot reuse an unfinished result')
    for filename, digest in record['artifact_sha256'].items():
        if fingerprint(filename) != digest:
            raise ValueError('Completed artifact changed: '+filename)
    run = Path(record['run_directory'])
    summary = read_json_file(run/'summary.json')
    if summary != record['summary'] or summary['dataset'] != job['dataset'] or summary['seed'] != job['seed']:
        raise ValueError('Result identity disagrees with the requested job')
    if summary['test_evaluations'] != 1 or not read_json_file(run/'status.json')['status'].startswith('completed_'):
        raise ValueError('Only completed, single-test evaluations may be reused')
    if (summary['max_epochs'] != 2000 or summary['patience'] != 100 or summary['lr'] != .05
            or summary['optimizer'] != 'Adam' or summary['training_batch_size'] != 0):
        raise ValueError('Head settings differ from the verified protocol')
    if job['dataset'] != 'ele-fashion':
        if summary.get('graph_policy') != job['policy'] or summary.get('split_id', 0) != job['split_id']:
            raise ValueError('Incorrect graph policy or split')
    directory = root/'data/processed'/('text_graphs' if job['dataset'] in ('wiki-cs', 'ogbn-arxiv') else 'mm_graph')/job['dataset']
    labels = np.load(directory/'labels.npy', mmap_mode='r')
    ids = (np.flatnonzero(np.load(directory/'test_mask_indices.npy')) if job['dataset'] == 'wiki-cs'
           else np.load(directory/'test_indices.npy'))
    with np.load(run/'test_predictions.npz') as output:
        np.testing.assert_array_equal(output['node_ids'], ids)
        np.testing.assert_array_equal(output['labels'], labels[ids])
        score = float(np.mean(output['prediction'] == output['labels']))
    if score != summary['test_accuracy']:
        raise ValueError('Saved predictions disagree with the claimed score')
    return {'official_test_nodes': len(ids), 'accuracy_recomputed': score, 'official_ids_and_labels_exact': True}


def import_confirmations(profile, jobs):
    model, _, _, full_pointer = reference_assets()
    model_hash = fingerprint(model)
    if model_hash != profile['encoder_sha256']:
        raise ValueError('Use the exact encoder behind the confirmed four results')
    fast_pointer = read_json_file(ROOT/'artifacts/workflow_state/fast_graph_screen_current.json')
    fast = read_json_file(fast_pointer['ledger'])
    if fast['state'] != 'completed_single_seed_graph_screen':
        raise ValueError('Finish the validation-locked confirmations first')
    full = read_json_file(full_pointer['ledger'])
    stability = None
    stability_pointer = ROOT/'artifacts/workflow_state/corrected_node_stability_current.json'
    if stability_pointer.exists():
        stability = read_json_file(read_json_file(stability_pointer)['ledger'])
        if stability['identity']['profile'] != profile or stability['identity']['encoder_sha256'] != model_hash:
            raise ValueError('Stability runs used a different frozen protocol')
    imported = {}
    for job in jobs:
        dataset = job['dataset']
        if dataset == 'ele-fashion':
            source = full['records'][f"eval_came_42_node_ele-fashion_0_{job['seed']}"]
            if fingerprint(source['summary']['encoder_checkpoint']) != model_hash:
                raise ValueError('Ele-fashion used a different encoder')
        elif job['seed'] == 0 and job['split_id'] == 0:
            lock = fast['selections'][dataset]
            if lock['policy'] != job['policy'] or not lock['locked_before_test'] or lock.get('test_used_for_selection', False):
                raise ValueError('No valid pre-test graph-policy lock')
            source = fast['records']['confirm_'+dataset]
            if source['summary']['encoder_sha256'] != model_hash:
                raise ValueError('Confirmation belongs to another encoder')
        elif stability and stability['records'].get(job['id'], {}).get('state') == 'completed':
            source = stability['records'][job['id']]
        else:
            continue
        record = copy.deepcopy(source)
        origin = (full_pointer['ledger'] if dataset == 'ele-fashion' else
                  fast_pointer['ledger'] if job['seed'] == 0 and job['split_id'] == 0 else
                  read_json_file(stability_pointer)['ledger'])
        record.update(job=job, reused=True, reused_from=str(origin))
        record['verification'] = verify_record(record, job)
        imported[job['id']] = record
    provenance = {'full_ledger': str(full_pointer['ledger']), 'fast_ledger': str(fast_pointer['ledger']),
                  'fast_ledger_sha256': fingerprint(fast_pointer['ledger']), 'selections': fast['selections']}
    return model, imported, provenance


def collect_artifacts(run):
    run = Path(run)
    files = [run/n for n in ('summary.json', 'manifest.json', 'config.json', 'status.json',
                            'test_predictions.npz', 'logs/metrics.csv', 'logs/metrics.jsonl')]
    files += list((run/'figures').glob('*.png')) + list(run.glob('validation_predictions.npz'))
    return {str(p): fingerprint(p) for p in files}


def report(ledger, directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    profile, jobs = ledger['identity']['profile'], ledger['identity']['jobs']
    rows, groups = [], []
    for dataset in profile['datasets']:
        expected = [j for j in jobs if j['dataset'] == dataset]
        scores = []
        for job in expected:
            record = ledger['records'].get(job['id'], {})
            if record.get('state') == 'completed':
                score = record['summary']['test_accuracy']*100
                scores.append(score)
                rows.append({'dataset': dataset, 'split_id': job['split_id'], 'seed': job['seed'],
                             'test_pct': score, 'reused': record.get('reused', False), 'run_directory': record['run_directory']})
        mean = float(np.mean(scores)) if scores else None
        groups.append({'dataset': dataset, 'completed': len(scores), 'expected': len(expected),
                       'complete_requested_grid': len(scores) == len(expected), 'mean_test_pct': mean,
                       'sample_sd_pp': float(np.std(scores, ddof=1)) if len(scores)>1 else None,
                       'paper_test_pct': profile['paper_test_pct'][dataset],
                       'gap_pp': mean-profile['paper_test_pct'][dataset] if mean is not None else None})
    atomic_write_json(directory/'results.json', {'state': ledger['state'], 'groups': groups, 'per_run': rows,
                       'scope': ledger['identity']['scope'], 'paper_reproduction_complete': False,
                       'statistics_note': 'Wiki-CS SD pools seeds and official splits; complete only at 400/400. Partial means are provisional.'})
    with (directory/'per_seed_results.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=['dataset','split_id','seed','test_pct','reused','run_directory'])
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4)
    ax.bar(x-.18, [g['mean_test_pct'] if g['mean_test_pct'] is not None else 0 for g in groups], .36, label='Observed mean (completed runs)')
    ax.bar(x+.18, [g['paper_test_pct'] for g in groups], .36, label='Paper mean (reference)')
    for index, group in enumerate(groups):
        ax.text(index, 92, f"{group['completed']}/{group['expected']}", ha='center')
    ax.set(xticks=x, xticklabels=profile['datasets'], ylim=(0,100), ylabel='Test accuracy (%)', title='Frozen four-task protocol: repeat coverage')
    ax.legend(loc='lower left'); fig.text(.02,.02,'Source: per_seed_results.csv. Counts show completed/requested. Partial means are not final; author details remain unresolved.', fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(directory/'four_task_comparison.png', dpi=160); plt.close(fig)


def child_arguments(job, profile, model, record):
    if job['dataset'] == 'ele-fashion':
        raise ValueError('Ele-fashion must only reuse its completed ten runs')
    args = ['-m','came_repro.quality_probe','--dataset',job['dataset'],'--representation','came',
            '--checkpoint',str(model),'--graph-policy',job['policy'],'--seed',str(job['seed']),
            '--split-id',str(job['split_id']),'--epochs',str(profile['probe_epochs']),
            '--patience',str(profile['probe_patience']),'--inference-batch-size',str(profile['inference_batches'][job['dataset']]),
            '--validation-cache-mib',str(profile['validation_cache_mib']), '--owned-cache-limit-gib',str(profile['owned_cache_limit_gib']), '--confirm-test']
    resume = best_resume(record, 'probe_last.pt')
    if resume:
        args += ['--resume-head',str(resume)]
    return args


def evict_completed_group(dataset, profile, model, ledger, directory):
    if dataset == 'ele-fashion':
        return
    group = [j for j in ledger['identity']['jobs'] if j['dataset'] == dataset]
    if not all(ledger['records'].get(j['id'], {}).get('state') == 'completed' for j in group):
        return
    from .quality_probe import owned_cache_path, remove_owned_cache
    storage = read_json_file(ROOT/'configs/storage.json')
    cache = owned_cache_path(storage, fingerprint(model), dataset, profile['graph_policies'][dataset])
    if cache.exists():
        saved = read_json_file(cache/'inference_state.json')
        if not saved['complete'] or saved['identity']['checkpoint_sha256'] != fingerprint(model):
            raise ValueError('Refusing to delete an incomplete or unrelated derived cache')
        remove_owned_cache(cache, storage)
        atomic_write_json(directory/(dataset+'_cache_eviction.json'), {'path':str(cache),'derived_only':True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/corrected_four_node_tasks.json')
    parser.add_argument('--single-seed', action='store_true', help='Show/reuse the four confirmed results, without new seeds')
    parser.add_argument('--stability-check', action='store_true', help='Predeclared seeds 0,1,2, Wiki split 0; partial grid')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Verify and save plan/results without launching training')
    parser.add_argument('--max-new-jobs', type=int, help='Explicit partial repeat budget, never a completed-paper claim')
    args = parser.parse_args()
    if args.max_new_jobs is not None and args.max_new_jobs < 1:
        parser.error('--max-new-jobs must be positive')
    profile = read_json_file(args.profile)
    jobs = build_jobs(profile, args.single_seed, args.stability_check)
    verify_originals()
    model, imported, parents = import_confirmations(profile, jobs)
    if args.dry_run:
        print(json.dumps({'planned_evaluations':len(jobs),'reused_confirmed':len(imported),
                          'pending':len(jobs)-len(imported),'additional_pretraining_updates':0,
                          'ele_fashion_retrained':False,'training_started':False,'graph_policies':profile['graph_policies']}, indent=2))
        return 0
    if Path(sys.executable).resolve() != (ROOT/'.venv-came39/Scripts/python.exe').resolve():
        raise RuntimeError('Project Python environment required')
    from .storage_budget import check_budget
    check_budget()
    sources = {str(p.relative_to(ROOT)):fingerprint(p) for pattern in ('src/came_repro/*.py','scripts/*.py') for p in ROOT.glob(pattern)}
    data = {str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'data/processed').glob('*/*/manifest.json')}
    import torch
    import dgl
    scope = 'confirmed_single_seed' if args.single_seed else 'three_seed_stability' if args.stability_check else 'four_task_paper_repeat_grid'
    identity = {'profile':profile,'jobs':jobs,'scope':scope,
                'sources':sources,'data':data,'storage':read_json_file(ROOT/'configs/storage.json'),'parents':parents,
                'encoder_sha256':fingerprint(model),'originals':verify_originals(),
                'environment':{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'dgl':dgl.__version__,'cuda':torch.version.cuda}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    state = ROOT/'artifacts/workflow_state'
    pointer = state/('corrected_node_confirmed_current.json' if args.single_seed else
                     'corrected_node_stability_current.json' if args.stability_check else 'corrected_node_suite_current.json')
    if pointer.exists() and read_json_file(pointer)['identity_key'] != key:
        raise ValueError('Source/data/config changed: retain and review the previous corrected ledger')
    directory = ROOT/'artifacts/corrected_node_suite'/key
    path, pause = state/('corrected_node_suite_'+key+'.json'), state/('corrected_node_suite_'+key+'_pause.json')
    lock = acquire_gpu_lock(ROOT/'.runtime/corrected_node_suite.lock')
    logger = logging.getLogger('corrected_node_suite.'+key)
    ledger = None
    try:
        gpu = acquire_gpu_lock(); gpu.close()
        directory.mkdir(parents=True, exist_ok=True)
        ledger = read_json_file(path) if path.exists() else {'identity':identity,'state':'ready','active_seconds':0.,'records':imported}
        for jid, rec in imported.items():
            existing = ledger['records'].get(jid)
            if not existing or existing['state'] != 'completed':
                if existing:
                    rec['superseded_local_attempts'] = existing.get('attempts', [])
                ledger['records'][jid] = rec
        for job in jobs:
            rec = ledger['records'].get(job['id'])
            if rec and rec['state'] == 'completed':
                verify_record(rec, job)
        atomic_write_json(pointer, {'identity_key':key,'ledger':str(path),'pause_file':str(pause),'directory':str(directory)})
        atomic_write_json(directory/'provenance.json', identity)
        atomic_write_json(directory/'resolved_profile.json',profile)
        atomic_write_json(directory/'plan.json',jobs)
        atomic_write_json(path,ledger)
        report(ledger,directory)
        if args.prepare_only:
            print('CORRECTED_PREPARED='+str(directory),flush=True)
            return 0
        if pause.exists():
            pause.unlink()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        logger.setLevel(logging.INFO)
        for handler in (logging.StreamHandler(), logging.FileHandler(directory/('pipeline_'+stamp+'.log'),encoding='utf-8')):
            fmt = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s'); fmt.converter=time.gmtime
            handler.setFormatter(fmt); logger.addHandler(handler)
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        ledger['state']='running'
        completed_new=0
        for index, job in enumerate(jobs):
            record=ledger['records'].setdefault(job['id'],{'state':'pending','job':job,'attempts':[]})
            if record['state']=='completed':
                continue
            if pause.exists() or (args.max_new_jobs and completed_new >= args.max_new_jobs):
                ledger['state']='paused'; break
            if any(fingerprint(ROOT/p)!=h for p,h in sources.items()) or read_json_file(args.profile)!=profile:
                raise ValueError('Code/profile changed during execution')
            check_budget()
            logger.info('CORRECTED_JOB %d/%d %s; frozen encoder, validation-selected head; Ele-fashion retained',index+1,len(jobs),job['id'])
            code,run=run_child(child_arguments(job,profile,model,record),pause,ledger,path,record,logger,float('inf'))
            if code!=0 or not run:
                record['state']='paused' if pause.exists() or code==75 else 'failed'
                ledger['state']=record['state']; break
            record.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_artifacts(run))
            record['verification']=verify_record(record,job)
            atomic_write_json(path,ledger); report(ledger,directory)
            evict_completed_group(job['dataset'],profile,model,ledger,directory)
            completed_new+=1
        else:
            ledger['state']='completed_confirmed_single_seed' if args.single_seed else 'completed_three_seed_stability' if args.stability_check else 'completed_four_task_repeat_grid'
        atomic_write_json(path,ledger); report(ledger,directory)
        print('CORRECTED_STATE='+ledger['state'],flush=True)
        print('CORRECTED_RESULTS='+str(directory/'results.json'),flush=True)
        return 1 if ledger['state']=='failed' else 0
    except BaseException as error:
        if ledger is not None:
            ledger['state']='paused' if isinstance(error,KeyboardInterrupt) else 'failed'
            ledger['error']=str(error); atomic_write_json(path,ledger); report(ledger,directory)
        raise
    finally:
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for handler in list(logger.handlers):
            handler.close(); logger.removeHandler(handler)
        lock.close()
