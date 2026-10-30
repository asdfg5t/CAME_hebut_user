"""Persistent full CAME task grid with coherent checkpoint resumes and honest 24h cap."""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid

from .author_loader import ROOT, fingerprint, verify_originals
from .ide_workflow import read_json, write_json, acquire_gpu_lock
from .variants import expected_loss_weights


def build_jobs(profile, training):
    if profile['evaluation_seeds'] != list(range(10)) or profile['wiki_node_seeds'] != list(range(20)) or profile['wiki_split_ids'] != list(range(20)):
        raise ValueError('Full downstream repeats and Wiki-CS grid cannot be reduced in this profile')
    if not profile['pretraining_seeds'] or len(set(profile['pretraining_seeds'])) != len(profile['pretraining_seeds']):
        raise ValueError('Explicit distinct pretraining seeds required')
    if profile['probe_epochs'] != 2000 or training['training']['max_steps'] is not None:
        raise ValueError('No reduced probe limit or pretraining step cap in the full grid')
    if len(set(profile['ablation_variants'])) != 9:
        raise ValueError('All nine ablation variants required')
    jobs = []
    def train(protocol, seed, variant='came'):
        name = 'pretrain_' + protocol + '_' + variant + '_seed' + str(seed)
        config = copy.deepcopy(training)
        config['name'] = 'suite_' + name
        config['seed'] = seed
        if protocol == 'few_shot':
            config['datasets'][1] = 'books-nc'
            config['sampling']['seed_limits']['books-nc'] = config['sampling']['seed_limits'].pop('books-lp')
        if variant != 'came':
            config['variant'] = variant
            config['model'].update(expected_loss_weights(config['model'], variant))
        config['deviations'].append('Full-suite grid; protocol=' + protocol + '; variant=' + variant + '; declared fixed seed pools retained.')
        from .joint_train import validate_joint
        validate_joint(config)
        jobs.append({'id': name, 'kind': 'pretraining', 'protocol': protocol, 'variant': variant, 'seed': seed, 'config': config})
        return name
    def evaluate(task, dataset, view, dependency=None, variant=None, pretrained_seed=None):
        splits = profile['wiki_split_ids'] if dataset == 'wiki-cs' and task == 'node' else [0]
        seeds = profile['wiki_node_seeds'] if dataset == 'wiki-cs' and task == 'node' else profile['evaluation_seeds']
        for split in splits:
            for seed in seeds:
                variant_name = variant or view
                name = '_'.join(map(str, ['eval', variant_name, pretrained_seed, task, dataset, split, seed]))
                jobs.append({'id': name, 'kind': 'evaluation', 'task': task, 'dataset': dataset, 'representation': view,
                             'variant': variant_name, 'pretraining_seed': pretrained_seed, 'seed': seed, 'split_id': split, 'dependency': dependency})
    main_tasks = [('node', d) for d in ('ele-fashion','wiki-cs','ogbn-arxiv','books-nc','ogbn-products')]
    main_tasks += [('link', d) for d in ('sports-copurchase','cloth-copurchase','books-lp')]
    main_tasks += [('relation', d) for d in ('FB15K237','WN18RR')]
    few_tasks = [('few_shot', d) for d in ('books-nc','ogbn-arxiv','wiki-cs')]
    for seed in profile['pretraining_seeds']:
        main = train('representation', seed)
        for task, dataset in main_tasks:
            evaluate(task, dataset, 'came', main, 'came', seed)
        few = train('few_shot', seed)
        for task, dataset in few_tasks:
            evaluate(task, dataset, 'came', few, 'came', seed)
    for view in ('clip_text','clip_image'):
        for task, dataset in main_tasks + few_tasks:
            if view == 'clip_image' and dataset not in ('ele-fashion','books-nc','sports-copurchase','cloth-copurchase','books-lp'):
                continue
            evaluate(task, dataset, view)
    for seed in profile['pretraining_seeds']:
        for variant in profile['ablation_variants']:
            dependency = train('representation', seed, variant)
            for task, dataset in (('node','ele-fashion'),('node','books-nc'),('link','cloth-copurchase'),('link','sports-copurchase')):
                evaluate(task, dataset, 'came', dependency, variant, seed)
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate experiment identity')
    return jobs


def job_arguments(job, profile, generated, records):
    if job['kind'] == 'pretraining':
        arguments = ['-m','came_repro.subset_offline_train','--config',str(generated / (job['id'] + '.json'))]
        resume_name, flag = 'last.pt', '--resume'
    else:
        module = 'came_repro.link_evaluation' if job['task'] == 'link' else 'came_repro.downstream'
        batch = profile['products_inference_batch_size'] if job['dataset'] == 'ogbn-products' else profile['inference_batch_size']
        arguments = ['-m',module,'--dataset',job['dataset'],'--representation',job['representation'],'--seed',str(job['seed']),
                     '--inference-batch-size',str(batch)]
        if job['dependency']:
            dependency = records[job['dependency']]
            if dependency['state'] != 'completed':
                raise ValueError('Incomplete encoder cannot unlock full-suite downstream experiments')
            arguments += ['--checkpoint',dependency['encoder_checkpoint']]
        if job['task'] == 'link':
            arguments += ['--epochs',str(profile['lp_epochs']),'--patience',str(profile['lp_patience'])]
            resume_name = 'link_head_last.pt'
        else:
            arguments += ['--task',job['task'],'--split-id',str(job['split_id']),'--epochs',str(profile['probe_epochs']),
                          '--patience',str(profile['probe_patience'])]
            resume_name = 'probe_last.pt' if job['task'] != 'few_shot' else None
        flag = '--resume-head'
    if resume_name:
        # Latest attempts before the first optimizer checkpoint may still be in inference;
        # retain the newest earlier coherent training checkpoint if there is one.
        for attempt in reversed(records.get(job['id'], {}).get('attempts', [])):
            run = attempt.get('run_directory')
            if run and (Path(run) / 'manifest.json').is_file():
                path = Path(read_json(Path(run) / 'manifest.json')['checkpoint_directory']) / resume_name
                if path.is_file():
                    arguments += [flag,str(path)]
                    break
    return arguments


def validate_completed(record):
    for path, expected in record.get('artifact_sha256', {}).items():
        if not Path(path).is_file() or fingerprint(path) != expected:
            raise ValueError('Completed artifact changed; refusing to skip or mix experiments: ' + path)


def evict_finished_embedding_cache(job, jobs, ledger, storage):
    """Delete only this suite's derived cache after every dependent seed has finished."""
    if job['kind'] != 'evaluation' or not job['dependency']:
        return
    group = [item for item in jobs if item['kind']=='evaluation' and item['dependency']==job['dependency'] and item['dataset']==job['dataset']]
    if not all(ledger['records'].get(item['id'], {}).get('state')=='completed' for item in group):
        return
    encoder = ledger['records'][job['dependency']]['encoder_checkpoint']
    digest = fingerprint(encoder)
    root = Path(storage['embedding_cache_root']).resolve()
    target = (root / digest[:20] / job['dataset']).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError('Embedding cache deletion escaped the configured cache root')
    if not target.exists():
        return
    state = read_json(target / 'inference_state.json')
    if state['identity']['checkpoint_sha256'] != digest or not state['complete']:
        raise ValueError('Only this suite encoder\'s completed derived cache can be evicted')
    ledger.setdefault('cache_evictions', []).append({'directory':str(target),'checkpoint_sha256':digest,
                    'embedding_sha256':state['output_sha256'],'dataset':job['dataset'],
                    'reason':'All dependent evaluations completed; raw data, results and checkpoints preserved'})
    shutil.rmtree(target)


def aggregate_completed(records, jobs):
    groups = {}
    for record in records.values():
        if record['state'] != 'completed' or record['job']['kind'] != 'evaluation':
            continue
        job, summary = record['job'], record['summary']
        key = (job['variant'], job['pretraining_seed'], job['task'], job['dataset'])
        values = {'accuracy_'+str(row['shot'])+'shot':row['accuracy'] for row in summary.get('episodes', [])}
        if 'test_accuracy' in summary:
            values['test_accuracy'] = summary['test_accuracy']
        values.update({'official_'+name:value for name,value in summary.get('test', {}).get('official', {}).items()})
        groups.setdefault(key, []).append({'seed':job['seed'],'split_id':job['split_id'],'metrics':values})
    output = []
    for key, rows in groups.items():
        expected = [job for job in jobs if job['kind']=='evaluation' and (job['variant'],job['pretraining_seed'],job['task'],job['dataset'])==key]
        per_split = {}
        for split in sorted({row['split_id'] for row in rows}):
            selected = [row for row in rows if row['split_id']==split]
            per_split[str(split)] = {name:{'mean':statistics.mean(row['metrics'][name] for row in selected),
                        'sample_std':statistics.stdev(row['metrics'][name] for row in selected) if len(selected)>1 else None,
                        'count':len(selected)} for name in selected[0]['metrics']}
        output.append({'variant':key[0],'pretraining_seed':key[1],'task':key[2],'dataset':key[3],
                       'completed_jobs':len(rows),'expected_jobs':len(expected),'complete_requested_grid':len(rows)==len(expected),
                       'per_split':per_split,'raw_metrics':rows,'metric_scale':'fractions; sample SD uses ddof=1; splits kept separate',
                       'paper_reproduction_complete':False})
    return output


def commit_completion(job, record, run_directory):
    run = Path(run_directory)
    summary, status = read_json(run / 'summary.json'), read_json(run / 'status.json')['status']
    paths = [run / 'summary.json', run / 'config.json', run / 'manifest.json', run / 'status.json']
    if job['kind'] == 'pretraining':
        if status != 'completed_subset_pretraining' or not summary.get('complete_declared_subset_pretraining') or summary['completed_subset_rounds'] != 8:
            raise ValueError('Partial pretraining cannot complete a full-suite encoder job')
        checkpoint = Path(read_json(run / 'manifest.json')['checkpoint_directory']) / 'encoder_final.pt'
        paths.append(checkpoint)
        record['encoder_checkpoint'] = str(checkpoint)
    elif status != 'completed_downstream_task':
        raise ValueError('Incomplete downstream result cannot be marked completed')
    paths += list(run.glob('test*.npz')) + list(run.glob('episode*.npz')) + list((run / 'figures').glob('*.png'))
    record.update(state='completed', run_directory=str(run), summary=summary,
                  artifact_sha256={str(path):fingerprint(path) for path in paths})


def run_child(arguments, pause_file, ledger, ledger_path, record, logger, limit_seconds):
    attempt = {'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'run_directory':None, 'arguments':arguments}
    record.setdefault('attempts', []).append(attempt)
    record['state'] = 'running'
    write_json(ledger_path, ledger)
    process = subprocess.Popen([sys.executable,'-u','-m','came_repro.suite_job','--',*arguments], cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                env={**os.environ,'CAME_PAUSE_FILE':str(pause_file),'CAME_SUITE_PARENT_PID':str(os.getpid()),'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'})
    output = queue.Queue()
    def read_output():
        try:
            for line in process.stdout:
                output.put(line.rstrip('\r\n'))
        finally:
            output.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started, last_saved, ended, forced, budget_pause_at = time.monotonic(), time.monotonic(), False, False, None
    charged = 0.
    def save_usage():
        nonlocal charged
        elapsed = time.monotonic() - started
        ledger['active_seconds'] += elapsed - charged
        charged = elapsed
        attempt['elapsed_seconds'] = elapsed
        write_json(ledger_path, ledger)
    def consume(line):
        if line:
            logger.info('%s',line)
            if line.startswith('RUN_DIR='):
                attempt['run_directory'] = line.removeprefix('RUN_DIR=')
                save_usage()
    try:
        while not ended:
            elapsed = time.monotonic() - started
            if elapsed >= limit_seconds and budget_pause_at is None:
                write_json(pause_file, {'reason':'cumulative_24h_budget','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                budget_pause_at = time.monotonic()
                logger.warning('Cumulative runtime limit reached; requesting a coherent pause, no completed-result claim.')
            if budget_pause_at and time.monotonic() - budget_pause_at > 30:
                forced = True
                process.terminate()
                break
            try:
                line = output.get(timeout=.5)
            except queue.Empty:
                line = ''
            if line is None:
                ended = True
            else:
                consume(line)
            if time.monotonic() - last_saved >= 1:
                save_usage()
                last_saved = time.monotonic()
        code = process.wait(timeout=10)
        reader.join(timeout=1)
        while not output.empty():
            consume(output.get_nowait())
        attempt.update(returncode=code, forced_budget_stop=forced)
        save_usage()
        return code, attempt['run_directory']
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        attempt['forced_interruption'] = True
        record['state'] = 'interrupted'
        save_usage()
        raise
    finally:
        if not reader.is_alive():
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,default=ROOT / 'configs/full_suite_resumable.json')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    profile = read_json(args.profile)
    training = read_json(ROOT / profile['training_config'])
    jobs = build_jobs(profile, training)
    counts = {'pretraining':sum(job['kind']=='pretraining' for job in jobs), 'evaluation':sum(job['kind']=='evaluation' for job in jobs)}
    if not 0 < profile['max_active_hours'] <= 24:
        raise ValueError('This profile limits cumulative active runtime to at most 24 hours')
    verify_originals()
    if args.dry_run:
        print(json.dumps({'jobs':len(jobs),'counts':counts,'profile':profile,'training_started':False,
                          'all_paper_experiments_implemented':False,'one_day_completion_guaranteed':False},indent=2))
        return
    if Path(sys.executable).resolve() != (ROOT / '.venv-came39/Scripts/python.exe').resolve():
        raise RuntimeError('Verified project environment required')
    lock = acquire_gpu_lock(ROOT / '.runtime/full_suite.lock')
    logger = None
    session, ledger, ledger_path = None, None, None
    try:
        # An orphaned child keeps this lock itself; never launch a second GPU job.
        check = acquire_gpu_lock()
        check.close()
        sources = {str(path.relative_to(ROOT)):fingerprint(path) for pattern in ('src/came_repro/*.py','scripts/*.py') for path in ROOT.glob(pattern)}
        data = {str(path.relative_to(ROOT)):fingerprint(path) for path in (ROOT / 'data/processed').glob('*/*/manifest.json')}
        import torch
        import dgl
        environment = {'executable':sys.executable,'python':sys.version,'torch':torch.__version__,'dgl':dgl.__version__,'cuda':torch.version.cuda}
        identity = {'profile':profile,'jobs':jobs,'sources':sources,'data_manifests':data,'storage':read_json(ROOT / 'configs/storage.json'),
                    'environment':environment,'originals':verify_originals()}
        key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
        state_directory = ROOT / 'artifacts/workflow_state'
        state_directory.mkdir(exist_ok=True,parents=True)
        pointer = state_directory / 'full_suite_current.json'
        ledger_path = state_directory / ('full_suite_' + key + '.json')
        if pointer.is_file() and read_json(pointer)['identity_key'] != key:
            raise ValueError('Source/data/protocol changed since this suite; retain the old ledger and review before starting a different suite')
        pause_file = state_directory / ('full_suite_' + key + '_pause.json')
        ledger = read_json(ledger_path) if ledger_path.is_file() else {'identity':identity,'active_seconds':0.,'records':{},'state':'ready'}
        for record in ledger['records'].values():
            if record['state'] == 'completed':
                validate_completed(record)
        if ledger['active_seconds'] >= profile['max_active_hours'] * 3600:
            print('BUDGET_EXHAUSTED: pending jobs retained; another Run does not reset the cumulative 24-hour cap.',flush=True)
            return
        if pause_file.is_file():
            pause_file.unlink()
        write_json(pointer,{'identity_key':key,'ledger':str(ledger_path),'pause_file':str(pause_file)})
        generated = ROOT / 'artifacts/resolved_suites' / key
        generated.mkdir(parents=True,exist_ok=True)
        for job in jobs:
            if job['kind'] == 'pretraining':
                path = generated / (job['id'] + '.json')
                if path.exists() and read_json(path) != job['config']:
                    raise ValueError('Resolved training configuration changed')
                write_json(path,job['config'])
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        session = ROOT / 'runs' / (stamp + '_full_suite_session_' + uuid.uuid4().hex[:6])
        (session / 'logs').mkdir(parents=True)
        logger = logging.getLogger(str(session))
        logger.setLevel(logging.INFO)
        for handler in (logging.StreamHandler(),logging.FileHandler(session / 'logs/pipeline.log',encoding='utf-8')):
            formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
            formatter.converter = time.gmtime
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        write_json(session / 'manifest.json',{'identity_key':key,'ledger':str(ledger_path),'identity':identity,'paper_reproduction_complete':False})
        write_json(session / 'config.json',profile)
        write_json(session / 'status.json',{'status':'running','paper_reproduction_complete':False})
        write_json(session / 'plan.json',jobs)
        for relative in sources:
            target = session / 'source' / relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT / relative,target)
        (ROOT / '.runtime').mkdir(exist_ok=True)
        write_json(ROOT / '.runtime/active_full_suite.json',{'pause_file':str(pause_file),'session':str(session),'pid':os.getpid()})
        ledger['state'] = 'running'
        write_json(ledger_path,ledger)
        # Every new session rechecks real CUDA/data/storage rather than trusting an old success.
        preflight_record = {}
        remaining = profile['max_active_hours'] * 3600 - ledger['active_seconds']
        code, _ = run_child([str(ROOT / 'scripts/preflight_reproduction.py')],pause_file,ledger,ledger_path,preflight_record,logger,remaining)
        if code != 0:
            ledger['state'] = 'paused' if pause_file.is_file() else 'failed_preflight'
        else:
            for index, job in enumerate(jobs):
                record = ledger['records'].setdefault(job['id'], {'state':'pending','job':job,'attempts':[]})
                if record['state'] == 'completed':
                    continue
                if pause_file.is_file() or ledger['active_seconds'] >= profile['max_active_hours'] * 3600:
                    ledger['state'] = 'paused_or_budget_exhausted'
                    break
                if any(fingerprint(ROOT / path) != value for path,value in sources.items()):
                    raise ValueError('Code changed during suite execution')
                logger.info('JOB %d/%d %s cumulative_hours=%.3f',index+1,len(jobs),job['id'],ledger['active_seconds']/3600)
                arguments = job_arguments(job,profile,generated,ledger['records'])
                remaining = profile['max_active_hours'] * 3600 - ledger['active_seconds']
                code, run = run_child(arguments,pause_file,ledger,ledger_path,record,logger,remaining)
                if code == 0 and run:
                    commit_completion(job,record,run)
                    write_json(ledger_path,ledger)
                    evict_finished_embedding_cache(job,jobs,ledger,identity['storage'])
                else:
                    record['state'] = 'paused' if pause_file.is_file() or code == 75 else 'failed'
                    ledger['state'] = record['state']
                    write_json(ledger_path,ledger)
                    break
                write_json(ledger_path,ledger)
            else:
                ledger['state'] = 'completed_came_task_grid'
        write_json(ledger_path,ledger)
        write_json(session / 'summary.json',{'state':ledger['state'],'completed_jobs':sum(r['state']=='completed' for r in ledger['records'].values()),
                   'planned_jobs':len(jobs),'cumulative_active_hours':ledger['active_seconds']/3600,'ledger':str(ledger_path),
                   'paper_reproduction_complete':False,'other_paper_analyses_and_author_details_unresolved':True})
        write_json(session / 'status.json',{'status':ledger['state'],'paper_reproduction_complete':False})
        completed = [record for record in ledger['records'].values() if record['state']=='completed' and record['job']['kind']=='evaluation']
        write_json(session / 'aggregate_results.json', aggregate_completed(ledger['records'], jobs))
        write_json(session / 'coverage_gaps.json', {'other_baseline_methods_not_retrained':['BGRL','DGI','GraphMAE2','GRACE','GCOPE','SAMGPT','FUG','UniGraph2'],
                   'analyses_not_implemented':['Fig3 t-SNE','Fig4 and Figs5-16 loss-weight sweeps','Tables8-9 full expert analysis','Table10 modality corruption','Table6 matching efficiency measurements'],
                   'author_details_unresolved':['Exact subset selection and size','Whether reported seeds independently repeat pretraining','Precise PPR and optimizer/head protocols'],
                   'all_paper_experiments_implemented':False})
        with (session / 'results.csv').open('w',newline='',encoding='utf-8') as output:
            fields = ['variant','pretraining_seed','task','dataset','seed','split_id','test_accuracy','test_mrr','accuracy_1shot','accuracy_3shot','accuracy_5shot']
            writer = csv.DictWriter(output,fieldnames=fields,extrasaction='ignore')
            writer.writeheader()
            for record in completed:
                summary = record['summary']
                episodes = {row['shot']:row['accuracy'] for row in summary.get('episodes',[])}
                writer.writerow({**record['job'],'test_accuracy':summary.get('test_accuracy',''),
                                 'test_mrr':summary.get('test',{}).get('official',{}).get('mrr',''),
                                 **{'accuracy_'+str(shot)+'shot':episodes.get(shot,'') for shot in (1,3,5)}})
        print('SUITE_SESSION=' + str(session),flush=True)
        print('SUITE_STATE=' + ledger['state'],flush=True)
        return 1 if ledger['state'] in ('failed', 'failed_preflight') else 0
    except BaseException as error:
        if ledger is not None and ledger_path is not None:
            ledger['state'] = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            write_json(ledger_path,ledger)
        if session is not None:
            write_json(session / 'status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                       'error':str(error),'paper_reproduction_complete':False})
        raise
    finally:
        lock.close()
        if logger:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


if __name__ == '__main__':
    raise SystemExit(main())
