"""Small, resumable quality screen -> one continuation round -> validation gate."""
import argparse
import copy
import ctypes
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

from .author_loader import ROOT, fingerprint, verify_originals
from .atomic_io import read_json_file, atomic_write_json
from .ide_workflow import acquire_gpu_lock
from .quality_protocol import validate_profile, select_graph_policies, validation_gate


def active_clock():
    # Windows monotonic wall time includes suspend. This OS counter excludes it.
    if os.name == 'nt':
        value = ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
            raise OSError('Cannot read active runtime clock')
        return value.value / 10**7
    return time.monotonic()


def reference_assets():
    pointer = read_json_file(ROOT/'artifacts/workflow_state/full_suite_current.json')
    ledger = read_json_file(pointer['ledger'])
    if ledger['state'] == 'running':
        raise RuntimeError('Use request_full_suite_pause.py and wait for SAFE_PAUSE first')
    record = ledger['records']['pretrain_representation_came_seed42']
    from .full_suite import validate_completed
    validate_completed(record)
    if record['summary'].get('completed_subset_rounds') != 8 or record['summary'].get('step') != 10240:
        raise ValueError('This continuation plan requires the recorded 10240-update eight-round parent')
    run = Path(record['run_directory'])
    directory = Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])
    model, continuation = directory/'encoder_final.pt', directory/'last.pt'
    if not model.is_file() or not continuation.is_file():
        raise FileNotFoundError('Both final encoder and model/optimizer/RNG checkpoint are required')
    return model, continuation, read_json_file(run/'config.json'), pointer


def remaining_first_seed_jobs(encoder):
    jobs = []
    for task, dataset in [('node','ogbn-products'),('link','sports-copurchase'),('link','cloth-copurchase'),
                          ('link','books-lp'),('relation','FB15K237'),('relation','WN18RR')]:
        jobs.append({'id':'first_'+dataset,'kind':'remaining_main','dataset':dataset,'task':task,'encoder':encoder,'seed':0})
    return jobs


def evict_first_seed_cache(job, checkpoint, directory):
    if job['dataset'] not in {j['dataset'] for j in remaining_first_seed_jobs('reference')}:
        raise ValueError('Refusing to evict a cache needed by the paused NC queue')
    storage=read_json_file(ROOT/'configs/storage.json')
    cache_root=Path(storage['embedding_cache_root']).resolve()
    digest=fingerprint(checkpoint)
    target=(cache_root/digest[:20]/job['dataset']).resolve()
    if not target.is_relative_to(cache_root) or len(target.relative_to(cache_root).parts)!=2:
        raise ValueError('Cache deletion escaped the expected two-level path')
    if target.is_dir():
        saved=read_json_file(target/'inference_state.json')
        if not saved['complete'] or saved['identity']['checkpoint_sha256']!=digest:
            raise ValueError('Refusing to remove an incomplete or unrelated cache')
        shutil.rmtree(target)
        atomic_write_json(directory/('cache_eviction_'+job['dataset']+'.json'),
                          {'directory':str(target),'checkpoint_sha256':digest,'raw_or_results_deleted':False})


def screen_jobs(profile):
    jobs = [{'id':'clip_'+dataset,'kind':'probe','dataset':dataset,'representation':'clip_text','policy':'original','encoder':'reference'}
            for dataset in profile['baseline_datasets']]
    jobs += [{'id':'reference_'+dataset,'kind':'probe','dataset':dataset,'representation':'came','policy':'original','encoder':'reference'}
             for dataset in profile['datasets']]
    jobs += [{'id':'graph_'+dataset,'kind':'probe','dataset':dataset,'representation':'came','policy':'bidirected','encoder':'reference'}
             for dataset in profile['graph_probe_datasets']]
    return jobs


def best_resume(record, filename):
    for attempt in reversed(record.get('attempts', [])):
        if attempt.get('run_directory'):
            run = Path(attempt['run_directory'])
            location = run/'checkpoints/location.json'
            if location.is_file():
                path = Path(read_json_file(location)['checkpoint_directory'])/filename
                if path.is_file():
                    return path


def run_child(arguments, pause_file, ledger, ledger_path, record, logger, remaining):
    attempt = {'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arguments':arguments,'run_directory':None}
    record.setdefault('attempts', []).append(attempt)
    record['state'] = 'running'
    atomic_write_json(ledger_path,ledger)
    process = subprocess.Popen([sys.executable,'-u','-m','came_repro.suite_job','--',*arguments],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
                env={**os.environ,'CAME_PAUSE_FILE':str(pause_file),'CAME_SUITE_PARENT_PID':str(os.getpid()),'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'})
    output = queue.Queue()
    def reader():
        try:
            for line in process.stdout:
                output.put(line.rstrip())
        finally:
            output.put(None)
    thread = threading.Thread(target=reader,daemon=True)
    thread.start()
    started, charged, saved, requested = active_clock(), 0., active_clock(), None
    def persist():
        nonlocal charged, saved
        elapsed = active_clock()-started
        ledger['active_seconds'] += elapsed-charged
        charged, saved = elapsed, active_clock()
        attempt['elapsed_active_seconds'] = elapsed
        atomic_write_json(ledger_path,ledger)
    try:
        while True:
            if active_clock()-started >= remaining and requested is None:
                atomic_write_json(pause_file,{'reason':'quality_pilot_active_budget'})
                requested = active_clock()
            if requested is not None and active_clock()-requested > 60:
                attempt['forced_budget_stop'] = True
                process.terminate()
                break
            try:
                line = output.get(timeout=.5)
            except queue.Empty:
                line = ''
            if line is None:
                break
            if line:
                logger.info('%s',line)
                if line.startswith('RUN_DIR='):
                    attempt['run_directory'] = line.removeprefix('RUN_DIR=')
            if active_clock()-saved >= 2:
                persist()
        code = process.wait(timeout=15)
        thread.join(timeout=1)
        attempt['returncode'] = code
        persist()
        return code, attempt['run_directory']
    finally:
        if process.poll() is None:
            atomic_write_json(pause_file,{'reason':'quality_controller_interrupted'})
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)
        thread.join(timeout=1)
        if not thread.is_alive():
            process.stdout.close()
        persist()


def summarize(ledger, directory):
    rows = []
    for jid, record in ledger['records'].items():
        if record['state']=='completed' and record['job']['kind'] in ('probe','remaining_main'):
            row = record['summary']
            rows.append({'job':jid,'dataset':row['dataset'],'seed':row['seed'],'graph_policy':row.get('graph_policy','original'),
                         'validation_accuracy':row.get('validation_accuracy'),'test_accuracy':row.get('test_accuracy'),
                         'validation_mrr':row.get('validation_mrr'),
                         'test_mrr':row.get('test',{}).get('official',{}).get('mrr'),
                         'test_evaluations':row.get('test_evaluations'), 'run_directory':record['run_directory']})
    atomic_write_json(directory/'quality_results.json',{'records':rows,'selection':ledger.get('selection'),
        'state':ledger['state'],'active_hours':ledger['active_seconds']/3600,'paper_reproduction_complete':False})
    import csv
    with (directory/'quality_results.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=['job','dataset','seed','graph_policy','validation_accuracy','test_accuracy','validation_mrr','test_mrr','test_evaluations','run_directory'])
        writer.writeheader()
        writer.writerows(rows)
    # A scientific figure with source values persisted above, never a validation
    # score plotted against the paper's test result.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axis = plt.subplots(figsize=(8,4.8))
    groups=[('clip_','CLIP-text'),('reference_','CAME reference'),('graph_','Bidirected diagnostic'),('pilot_','One extra subset round')]
    domains=ledger['identity']['profile']['datasets']
    for i,(prefix,label) in enumerate(groups):
        values=[]
        for dataset in domains:
            r=ledger['records'].get(prefix+dataset,{})
            values.append(100*r['summary']['validation_accuracy'] if r.get('state')=='completed' else np.nan)
        axis.bar(np.arange(len(domains))+(i-1.5)*.18,values,.18,label=label)
    axis.set(xticks=np.arange(len(domains)),xticklabels=domains,ylim=(0,100),ylabel='Official validation accuracy (%)',title='Quality screen: one fixed seed; Wiki-CS split 0')
    axis.legend(fontsize=8)
    fig.text(.02,.01,'Source: quality_results.json and recorded run summaries. Missing bars = pending. Test does not select candidates.',fontsize=7)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(directory/'validation_comparison.png',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,default=ROOT/'configs/quality_pilot.json')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-jobs',type=int,help='Pause after this many additional completed jobs; protocol unchanged')
    parser.add_argument('--remaining-tasks',action='store_true',help='After quality review, run the six remaining main tasks at seed zero')
    args = parser.parse_args()
    profile = read_json_file(args.profile)
    validate_profile(profile)
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError('max-jobs must be positive')
    model, continuation, training, original_pointer = reference_assets()
    training = copy.deepcopy(training)
    training['name'] = 'quality_extra_subset_round'
    training['training']['epochs'] = 9
    training['training']['max_steps'] = None
    training['deviations'] = ['Parent run: '+note for note in training['deviations']]+['Quality diagnostic: one additional fixed subset round after the completed eight-round model; nine rounds are not paper eight epochs; no cumulative runtime deadline.']
    if args.dry_run:
        print(json.dumps({'screening_jobs':screen_jobs(profile),'additional_subset_rounds':1,
                          'added_optimizer_updates':1280,'pilot_validation_jobs':3,'confirmation_jobs_max':3,
                          'remaining_main_jobs':remaining_first_seed_jobs('validation_selected_encoder'),
                          'max_active_hours':profile['max_active_hours'],'test_used_for_selection':False,
                          'training_started':False,'reference_checkpoint':str(model)},indent=2))
        return 0
    if Path(sys.executable).resolve() != (ROOT/'.venv-came39/Scripts/python.exe').resolve():
        raise RuntimeError('Project Python environment required')
    lock = acquire_gpu_lock(ROOT/'.runtime/quality_pilot.lock')
    logger = None
    try:
        gpu = acquire_gpu_lock()
        gpu.close()
        verify_originals()
        from .storage_budget import check_budget
        check_budget()
        sources = {str(p.relative_to(ROOT)):fingerprint(p) for pattern in ('src/came_repro/*.py','scripts/*.py') for p in ROOT.glob(pattern)}
        data = {str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'data/processed').glob('*/*/manifest.json')}
        import torch
        import dgl
        identity = {'profile':profile,'sources':sources,'data':data,'reference_encoder_sha256':fingerprint(model),
                    'environment':{'executable':sys.executable,'python':sys.version,'torch':torch.__version__,'dgl':dgl.__version__,'cuda':torch.version.cuda},
                    'continuation_sha256':fingerprint(continuation),'training':training,'originals':verify_originals()}
        key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
        state = ROOT/'artifacts/workflow_state'
        directory = ROOT/'artifacts/quality_pilot'/key
        directory.mkdir(parents=True,exist_ok=True)
        pointer_path = state/'quality_pilot_current.json'
        if pointer_path.is_file() and read_json_file(pointer_path)['identity_key'] != key:
            raise ValueError('Quality protocol/source/data changed; preserve and review the old ledger before a new pilot')
        path, pause = state/('quality_pilot_'+key+'.json'), state/('quality_pilot_'+key+'_pause.json')
        ledger = read_json_file(path) if path.is_file() else {'identity':identity,'state':'ready','active_seconds':0.,'records':{}}
        if args.remaining_tasks and not (directory/'selection_lock.json').exists():
            raise RuntimeError('Run and review run_quality_pilot.py before the remaining single-seed tasks')
        for record in ledger['records'].values():
            if record['state']=='completed':
                for filename, digest in record['artifact_sha256'].items():
                    if fingerprint(filename) != digest:
                        raise ValueError('A completed diagnostic artifact changed')
        limit_seconds = float('inf') if profile['max_active_hours'] is None else profile['max_active_hours']*3600
        if ledger['active_seconds'] >= limit_seconds:
            print('QUALITY_BUDGET_EXHAUSTED: saved progress retained; Run does not reset the budget')
            return 0
        if pause.exists():
            pause.unlink()
        atomic_write_json(pointer_path,{'identity_key':key,'ledger':str(path),'pause_file':str(pause),'directory':str(directory)})
        atomic_write_json(directory/'resolved_profile.json',profile)
        atomic_write_json(directory/'extension_training.json',training)
        atomic_write_json(directory/'provenance.json',identity)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log_path = directory/('pipeline_'+stamp+'.log')
        logger = logging.getLogger(str(log_path))
        logger.setLevel(logging.INFO)
        for handler in (logging.StreamHandler(),logging.FileHandler(log_path,encoding='utf-8')):
            formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
            formatter.converter = time.gmtime
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        ledger['state'] = 'running'
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        completed_now = 0
        def execute(job):
            nonlocal completed_now
            record = ledger['records'].setdefault(job['id'],{'job':job,'state':'pending','attempts':[]})
            if record['state']=='completed':
                return True
            if pause.is_file() or ledger['active_seconds'] >= limit_seconds or (args.max_jobs and completed_now >= args.max_jobs):
                ledger['state'] = 'paused'
                return False
            if any(fingerprint(ROOT/name)!=digest for name,digest in sources.items()):
                raise ValueError('Source changed during quality pipeline execution')
            if job['id'].startswith('reference_') or (job.get('confirm') and job['encoder']=='reference' and job['policy']=='original'):
                previous = read_json_file(original_pointer['ledger'])
                jid = f"eval_came_42_node_{job['dataset']}_{profile['wiki_split_id'] if job['dataset']=='wiki-cs' else 0}_0"
                prior = previous['records'].get(jid)
                if prior and prior['state']=='completed':
                    from .full_suite import validate_completed
                    validate_completed(prior)
                    record.update(state='completed',summary=copy.deepcopy(prior['summary']),run_directory=prior['run_directory'],
                                  artifact_sha256=prior['artifact_sha256'],reused_completed_reference=True)
                    record['summary']['graph_policy'] = 'original'
                    record['summary']['test_used_for_current_selection'] = False
                    record['summary']['reference_test_previously_evaluated'] = True
                    if not job.get('confirm'):
                        record['summary']['test_accuracy'] = None
                        record['summary']['test_loss'] = None
                    atomic_write_json(path,ledger)
                    logger.info('Reused completed reference %s; no retraining and no test metric used for selection',jid)
                    summarize(ledger,directory)
                    return True
            if job['kind']=='pretraining':
                resume = best_resume(record,'last.pt') or continuation
                arguments = ['-m','came_repro.quality_pretrain','--config',str(directory/'extension_training.json'),'--resume',str(resume)]
            elif job['kind']=='remaining_main':
                checkpoint = model if job['encoder']=='reference' else Path(ledger['records']['extra_round']['encoder'])
                module = 'came_repro.link_evaluation' if job['task']=='link' else 'came_repro.downstream'
                arguments = ['-m',module,'--dataset',job['dataset'],'--representation','came','--seed','0','--checkpoint',str(checkpoint),
                             '--inference-batch-size','4' if job['dataset']=='ogbn-products' else '32']
                if job['task']=='link':
                    arguments += ['--epochs','100','--patience','10']
                    resume = best_resume(record,'link_head_last.pt')
                else:
                    arguments += ['--task',job['task'],'--epochs','2000','--patience','100']
                    resume = best_resume(record,'probe_last.pt')
                if resume:
                    arguments += ['--resume-head',str(resume)]
            else:
                arguments = ['-m','came_repro.quality_probe','--dataset',job['dataset'],'--representation',job['representation'],
                  '--graph-policy',job['policy'],'--seed',str(job.get('seed',profile['screen_seed'])),
                  '--split-id',str(profile['wiki_split_id'] if job['dataset']=='wiki-cs' else 0),
                  '--epochs',str(profile['probe_epochs']),'--patience',str(profile['probe_patience']),
                  '--inference-batch-size',str(profile['books_bidirected_inference_batch_size'] if job['dataset']=='books-nc' and job['policy']=='bidirected' else profile['inference_batch_size']),
                  '--validation-cache-mib',str(profile['validation_cache_mib']),'--owned-cache-limit-gib',str(profile['owned_embedding_limit_gib'])]
                if job['representation']=='came':
                    arguments += ['--checkpoint',str(model if job['encoder']=='reference' else Path(ledger['records']['extra_round']['encoder']))]
                resume = best_resume(record,'probe_last.pt')
                if resume:
                    arguments += ['--resume-head',str(resume)]
                if job.get('confirm'):
                    arguments += ['--confirm-test']
                if job.get('discard',True):
                    arguments += ['--discard-derived-cache']
            logger.info('QUALITY_JOB=%s completed_new=%d active_hours=%.3f',job['id'],completed_now,ledger['active_seconds']/3600)
            code, run = run_child(arguments,pause,ledger,path,record,logger,limit_seconds-ledger['active_seconds'])
            if code != 0 or not run:
                record['state'] = 'paused' if pause.exists() or code==75 else 'failed'
                ledger['state'] = record['state']
                return False
            run = Path(run)
            summary = read_json_file(run/'summary.json')
            status = read_json_file(run/'status.json')['status']
            if job['kind']=='pretraining':
                if status != 'completed_subset_pretraining' or summary['completed_subset_rounds']!=9 or summary['step']!=11520:
                    raise ValueError('Additional round did not finish the declared 1280 updates')
                location = Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])/'encoder_final.pt'
                record['encoder'] = str(location)
                artifacts = [location,run/'summary.json',run/'config.json',run/'manifest.json',run/'status.json']
            elif job['kind']=='remaining_main':
                if not status.startswith('completed_'):
                    raise ValueError('Remaining main task did not complete')
                artifacts = [run/n for n in ('summary.json','config.json','manifest.json','status.json')]
            else:
                if status!='completed_quality_probe' or summary['epochs_executed']<1 or summary['test_evaluations']!=int(bool(job.get('confirm'))):
                    raise ValueError('Unexpected diagnostic completion or test access')
                artifacts = [run/n for n in ('summary.json','config.json','manifest.json','status.json','logs/metrics.csv','logs/metrics.jsonl','figures/probe_curves.png')]
                artifacts += [run/('test_predictions.npz' if job.get('confirm') else 'selection_predictions.npz')]
            record.update(state='completed',summary=summary,run_directory=str(run),artifact_sha256={str(p):fingerprint(p) for p in artifacts})
            completed_now += 1
            atomic_write_json(path,ledger)
            summarize(ledger,directory)
            if job['kind']=='remaining_main':
                evict_first_seed_cache(job,checkpoint,directory)
            return True
        def finish():
            atomic_write_json(path,ledger)
            summarize(ledger,directory)
            print('QUALITY_STATE='+ledger['state'],flush=True)
            print('QUALITY_RESULTS='+str(directory/'quality_results.json'),flush=True)
            return 1 if ledger['state']=='failed' else 0
        for job in screen_jobs(profile):
            if not execute(job):
                return finish()
        reference = {d:ledger['records']['reference_'+d]['summary'] for d in profile['datasets']}
        probes = {d:ledger['records']['graph_'+d]['summary'] for d in profile['graph_probe_datasets']}
        policies = select_graph_policies(reference,probes,profile['min_graph_validation_gain_pp'])
        atomic_write_json(directory/'graph_selection.json',{'policies':policies,'selection_data':'validation only'})
        adjusted = {d:probes[d] if policies[d]=='bidirected' else reference[d] for d in profile['datasets']}
        if not execute({'id':'extra_round','kind':'pretraining'}):
            return finish()
        for dataset in profile['datasets']:
            job = {'id':'pilot_'+dataset,'kind':'probe','dataset':dataset,'representation':'came','policy':policies[dataset],'encoder':'pilot'}
            if not execute(job):
                return finish()
        candidate = {d:ledger['records']['pilot_'+d]['summary'] for d in profile['datasets']}
        gate = validation_gate(adjusted,candidate,profile)
        ledger['selection'] = {**gate,'graph_policies':policies,'encoder':'pilot' if gate['passed'] else 'reference',
                               'locked_before_confirmation':True}
        graph_gate = validation_gate(reference,adjusted,profile)
        ledger['selection']['stable_validation_gain'] = gate['passed'] or graph_gate['passed']
        # This file is frozen before any candidate confirmation test is evaluated.
        lock_path = directory/'selection_lock.json'
        if lock_path.exists() and read_json_file(lock_path)!=ledger['selection']:
            raise ValueError('Refusing to change a selection after confirmation began')
        atomic_write_json(lock_path,ledger['selection'])
        for dataset in profile['datasets'] if ledger['selection']['stable_validation_gain'] else []:
            for seed in profile['confirmation_seeds']:
                job = {'id':f'confirm_{dataset}_{seed}','kind':'probe','dataset':dataset,'representation':'came',
                       'policy':policies[dataset],'encoder':ledger['selection']['encoder'],'seed':seed,'confirm':True,
                       'discard':seed==profile['confirmation_seeds'][-1]}
                if not execute(job):
                    return finish()
        if profile['run_remaining_first_seed_tasks'] or args.remaining_tasks:
            for job in remaining_first_seed_jobs(ledger['selection']['encoder']):
                if not execute(job):
                    return finish()
        ledger['state'] = ('completed_first_seed_grid' if ledger['selection']['stable_validation_gain'] else 'completed_first_seed_grid_needs_quality_review') if args.remaining_tasks else ('completed_small_quality_confirmation' if ledger['selection']['stable_validation_gain'] else 'needs_protocol_review_no_stable_validation_gain')
        return finish()
    except BaseException as error:
        if 'ledger' in locals() and 'path' in locals():
            ledger['state'] = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            ledger['error'] = str(error)
            atomic_write_json(path,ledger)
        if logger:
            logger.exception('Quality pipeline stopped; completed results and checkpoints preserved')
        raise
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if logger:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
