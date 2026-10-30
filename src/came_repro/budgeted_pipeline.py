"""Same-method limited-budget experiments, isolated from the full running pipeline."""
import argparse
import csv
import datetime
import gc
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
import uuid

from .author_loader import ROOT, fingerprint, verify_originals
from .ide_workflow import CORPORA, acquire_gpu_lock, read_json, write_json


def validate_budget(config):
    if config['schema'] != 'came_limited_budget_v1' or not 0 < config['total_wall_hours'] <= 48:
        raise ValueError('An explicit limited budget of at most 48 hours is required')
    for key in ('representation_max_updates', 'few_shot_max_updates', 'probe_max_epochs', 'lp_max_epochs',
                'inference_batch_size', 'products_inference_batch_size', 'max_job_wall_minutes'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Positive integer required: ' + key)


def task_plan(config):
    tasks = [('node', 'ele-fashion', 'representation'), ('node', 'wiki-cs', 'representation'),
             ('node', 'ogbn-arxiv', 'representation'), ('link', 'sports-copurchase', 'representation'),
             ('relation', 'FB15K237', 'representation'), ('few_shot', 'wiki-cs', 'few_shot'),
             ('few_shot', 'ogbn-arxiv', 'few_shot')]
    if config['include_large_domains']:
        tasks += [('relation', 'WN18RR', 'representation'), ('node', 'books-nc', 'representation'),
                  ('few_shot', 'books-nc', 'few_shot'), ('link', 'cloth-copurchase', 'representation'),
                  ('link', 'books-lp', 'representation'), ('node', 'ogbn-products', 'representation')]
    return tasks


def evaluation_arguments(config, task, dataset, representation, checkpoint=None):
    module = 'came_repro.link_evaluation' if task == 'link' else 'came_repro.downstream'
    args = ['-m', module, '--dataset', dataset, '--representation', representation,
            '--seed', str(config['evaluation_seed'])]
    if checkpoint is not None:
        args += ['--checkpoint', str(checkpoint)]
    batch = config['products_inference_batch_size'] if dataset == 'ogbn-products' else config['inference_batch_size']
    args += ['--inference-batch-size', str(batch)]
    if task == 'link':
        args += ['--epochs', str(config['lp_max_epochs']), '--patience', str(config['lp_patience'])]
    else:
        args += ['--task', task, '--split-id', str(config['wiki_split_id']),
                 '--epochs', str(config['probe_max_epochs']), '--patience', str(config['probe_patience'])]
    return args


def newest_checkpoint(protocol, seed, before=None, variant='came'):
    candidates = []
    for run in (ROOT / 'runs').glob('*'):
        if before is not None and run.name in before:
            continue
        if not (run / 'config.json').is_file() or not (run / 'manifest.json').is_file():
            continue
        config, manifest = read_json(run / 'config.json'), read_json(run / 'manifest.json')
        if (config.get('datasets') != CORPORA[protocol] or config.get('seed') != seed
                or config.get('variant', 'came') != variant or 'evaluation' in config):
            continue
        checkpoint = Path(manifest['checkpoint_directory']) / 'last.pt'
        if checkpoint.is_file():
            candidates.append((checkpoint.stat().st_mtime_ns, checkpoint, run))
    return max(candidates) if candidates else None


def run_bounded(arguments, seconds, logger):
    """Read asynchronously so file hashing/inference silence cannot defeat the deadline."""
    if seconds <= 0:
        return {'returncode': None, 'timed_out': True, 'run_directory': None, 'cuda_oom': False}
    process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=str(ROOT), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                               env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'})
    output = queue.Queue()
    def read_output():
        try:
            for line in process.stdout:
                output.put(line.rstrip('\r\n'))
        finally:
            output.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline, ended, timed_out = time.monotonic() + seconds, False, False
    run, cuda_oom = None, False
    def consume(line):
        nonlocal run, cuda_oom
        if line is None:
            return
        logger.info('%s', line)
        if line.startswith('RUN_DIR='):
            run = line.removeprefix('RUN_DIR=')
        if 'out of memory' in line.lower() and ('cuda' in line.lower() or 'torch.OutOfMemoryError' in line):
            cuda_oom = True
    try:
        while not ended:
            if time.monotonic() >= deadline:
                timed_out = True
                logger.warning('Explicit time budget reached; keeping the latest consistent child checkpoint/cache.')
                if process.poll() is None:
                    process.terminate()
                break
            try:
                line = output.get(timeout=min(1., max(.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                ended = True
            else:
                consume(line)
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
        reader.join(timeout=1)
        while not output.empty():
            consume(output.get_nowait())
        return {'returncode': returncode, 'timed_out': timed_out, 'run_directory': run, 'cuda_oom': cuda_oom}
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if not reader.is_alive():
            process.stdout.close()


def result_row(job, outcome):
    row = {**job, 'status': 'time_budget_reached' if outcome['timed_out'] else
           'completed' if outcome['returncode'] == 0 else 'failed', 'run_directory': outcome['run_directory']}
    if outcome['run_directory'] and (Path(outcome['run_directory']) / 'summary.json').is_file():
        summary = read_json(Path(outcome['run_directory']) / 'summary.json')
        row['summary_sha256'] = fingerprint(Path(outcome['run_directory']) / 'summary.json')
        row['summary'] = summary
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/budgeted_24h.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = read_json(args.profile)
    validate_budget(config)
    expected = ROOT / '.venv-came39/Scripts/python.exe'
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError('Use the verified project environment')
    verify_originals()
    if args.dry_run:
        print(json.dumps({'profile': config, 'tasks': task_plan(config), 'model_changed': False,
                          'paper_epochs_completed_by_step_cap': False, 'training_started': False}, indent=2))
        return
    gpu_lock = acquire_gpu_lock()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = ROOT / 'runs' / (stamp + '_' + config['name'] + '_' + uuid.uuid4().hex[:6])
    (directory / 'logs').mkdir(parents=True)
    logger = logging.getLogger(str(directory))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
    formatter.converter = time.gmtime
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / 'logs/budget_pipeline.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    started = time.monotonic()
    deadline = started + config['total_wall_hours'] * 3600
    records, encoders = [], {}
    write_json(directory / 'config.json', config)
    sources = {}
    for pattern in ('src/came_repro/*.py', 'scripts/*.py', 'configs/**/*.json'):
        for path in ROOT.glob(pattern):
            target = directory / 'source' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            sources[str(path.relative_to(ROOT))] = fingerprint(path)
    write_json(directory / 'manifest.json', {'original_files': verify_originals(), 'sources_sha256': sources,
                                           'executable': sys.executable, 'scope': 'same_method_reduced_budget',
                                           'seed_repeats': 1, 'wiki_splits': 1, 'paper_reproduction_complete': False})
    def remaining():
        return max(0., deadline - time.monotonic())
    def verify_sources():
        if any(fingerprint(ROOT / path) != value for path, value in sources.items()):
            raise ValueError('Code/configuration changed during the limited-budget experiment; refusing mixed protocols')
    def save():
        write_json(directory / 'summary.json', {'profile': config, 'records': records,
                   'encoders': {name: str(value) for name, value in encoders.items()},
                   'elapsed_hours': (time.monotonic() - started) / 3600, 'paper_reproduction_complete': False,
                   'all_planned_jobs_successful': bool(records) and all(row['status'] == 'completed' for row in records),
                   'scope': 'Reduced updates/head epochs/seed repeats; partial pretraining, not eight full paper epochs.'})
        with (directory / 'results.csv').open('w', newline='', encoding='utf-8') as output:
            fields = ['task', 'dataset', 'representation', 'ablation_variant', 'status', 'run_directory', 'test_accuracy', 'test_mrr', 'test_hits10', 'accuracy_1shot', 'accuracy_3shot', 'accuracy_5shot']
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in records:
                summary = row.get('summary', {})
                ranking = summary.get('test', {}).get('official', {})
                episodes = {record['shot']: record['accuracy'] for record in summary.get('episodes', [])}
                writer.writerow({**row, 'test_accuracy': summary.get('test_accuracy', ''),
                                 'test_mrr': ranking.get('mrr', ''), 'test_hits10': ranking.get('hits@10', ''),
                                 **{'accuracy_' + str(shot) + 'shot': episodes.get(shot, '') for shot in (1, 3, 5)}})
    def evaluate(task, dataset, protocol, representation='came', checkpoint=None, ablation_variant=None):
        verify_sources()
        job = {'task': task, 'dataset': dataset, 'representation': representation, 'seed': config['evaluation_seed']}
        if ablation_variant:
            job['ablation_variant'] = ablation_variant
        if remaining() <= 0 or representation == 'came' and checkpoint is None:
            records.append({**job, 'status': 'deferred', 'reason': 'Time budget or valid encoder unavailable', 'run_directory': None})
            save()
            return
        arguments = evaluation_arguments(config, task, dataset, representation, checkpoint)
        limit = config['products_max_job_wall_minutes'] if dataset == 'ogbn-products' else config['max_job_wall_minutes']
        logger.info('EVALUATION %s; head budgets and one-seed adaptation are explicit.', job)
        outcome = run_bounded(arguments, min(remaining(), limit * 60), logger)
        records.append(result_row(job, outcome))
        if outcome['cuda_oom'] and remaining() > 0 and representation == 'came':
            arguments[arguments.index('--inference-batch-size') + 1] = '1'
            logger.warning('Explicit exact-inference retry with target batch=1; no neighbor/width/precision changes.')
            retry = run_bounded(arguments, min(remaining(), limit * 60), logger)
            records.append(result_row({**job, 'retry': 'exact_inference_batch_one'}, retry))
        save()
    try:
        from .storage_budget import check_budget
        check_budget(additional_cache_bytes=2 * 2**30)
        logger.warning('LIMITED-BUDGET REPRODUCTION: unchanged main model/method, reduced updates/repeats/head budgets; no full-paper claim.')
        preflight = run_bounded([str(ROOT / 'scripts/preflight_reproduction.py')], min(remaining(), 1800), logger)
        if preflight['returncode'] != 0:
            raise RuntimeError('CUDA/data/storage preflight did not pass')
        import torch
        from .joint_train import signature
        for protocol in ('representation', 'few_shot'):
            verify_sources()
            profile = ROOT / 'configs' / ('joint_' + protocol + '_reconstruction.json')
            base = read_json(profile)
            base['storage'] = read_json(ROOT / 'configs/storage.json')
            base['seed'] = config['pretraining_seed']
            candidate = newest_checkpoint(protocol, config['pretraining_seed']) if config['reuse_existing_checkpoints'] else None
            resume, prior_steps = None, 0
            if candidate:
                state = torch.load(candidate[1], map_location='cpu', weights_only=False)
                if state.get('schema') == 'came_joint_v1' and signature(state['config']) == signature(base):
                    resume, prior_steps = candidate[1], state['step']
                    logger.info('Reusing consistent %s checkpoint: %s; step=%d, parent_sha256=%s', protocol, resume, prior_steps, fingerprint(resume))
                del state
                gc.collect()
            max_updates = config[protocol + '_max_updates']
            if resume and prior_steps >= max_updates:
                encoders[protocol] = resume
                records.append({'task': 'pretraining', 'dataset': protocol, 'representation': 'came', 'status': 'completed',
                                'run_directory': str(candidate[2]), 'actual_checkpoint_steps': prior_steps,
                                'reason': 'Existing consistent checkpoint already reaches requested update budget; no extra updates.'})
                save()
                continue
            before = {path.name for path in (ROOT / 'runs').iterdir() if path.is_dir()}
            arguments = ['-m', 'came_repro.joint_train', '--config', str(profile), '--seed', str(config['pretraining_seed']), '--max-steps', str(max_updates)]
            if resume:
                arguments += ['--resume', str(resume)]
            logger.info('PRETRAINING %s target cumulative updates=%d; full eight-epoch coverage is not claimed.', protocol, max_updates)
            outcome = run_bounded(arguments, min(remaining(), config[protocol + '_wall_hours'] * 3600), logger)
            records.append(result_row({'task': 'pretraining', 'dataset': protocol, 'representation': 'came'}, outcome))
            own = newest_checkpoint(protocol, config['pretraining_seed'], before=before)
            if own:
                encoders[protocol] = own[1]
                records[-1]['checkpoint_sha256'] = fingerprint(own[1])
            elif resume:
                encoders[protocol] = resume
                records[-1]['encoder_fallback'] = 'Previous valid checkpoint retained; this phase did not save a newer one.'
            save()
        for task, dataset, protocol in task_plan(config):
            evaluate(task, dataset, protocol, checkpoint=encoders.get(protocol))
        if config['include_clip_baselines']:
            for view in ('clip_text', 'clip_image'):
                for task, dataset, protocol in task_plan(config):
                    if view == 'clip_image' and dataset not in ('ele-fashion', 'books-nc', 'sports-copurchase', 'cloth-copurchase', 'books-lp'):
                        continue
                    evaluate(task, dataset, protocol, representation=view)
        for variant in config['ablation_variants']:
            verify_sources()
            job = {'task': 'ablation_pretraining', 'dataset': variant, 'representation': 'came'}
            if remaining() <= 0:
                records.append({**job, 'status': 'deferred', 'run_directory': None})
                continue
            before = {path.name for path in (ROOT / 'runs').iterdir() if path.is_dir()}
            arguments = ['-m', 'came_repro.joint_train', '--config', str(ROOT / 'configs/ablations' / (variant + '.json')),
                         '--seed', str(config['pretraining_seed']), '--max-steps', str(config['ablation_max_updates'])]
            outcome = run_bounded(arguments, min(remaining(), config['ablation_wall_hours'] * 3600), logger)
            records.append(result_row(job, outcome))
            own = newest_checkpoint('representation', config['pretraining_seed'], before=before, variant=variant)
            if own:
                evaluate('node', 'ele-fashion', 'representation', checkpoint=own[1], ablation_variant=variant)
        deferred = ['Seven remaining ablation settings beyond the two selected variants', 'Nine extra downstream seeds and nineteen extra Wiki-CS splits']
        if not config['include_large_domains']:
            deferred.append('Large-domain downstream jobs omitted by the 12-hour preset; their raw data and pretraining corpus remain intact.')
        write_json(directory / 'omitted_scope.json', {'omitted': deferred, 'raw_data_deleted': False})
        save()
        write_json(directory / 'status.json', {'status': 'completed_limited_budget_with_recorded_outcomes',
                                             'paper_reproduction_complete': False, 'elapsed_hours': (time.monotonic() - started) / 3600})
        logger.info('BUDGET_RUN_DIR=%s; inspect successful/failed/deferred rows, never fill in missing metrics.', directory)
    except BaseException as error:
        logger.exception('Limited-budget pipeline stopped')
        save()
        write_json(directory / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'error': str(error)})
        raise
    finally:
        gpu_lock.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == '__main__':
    main()
