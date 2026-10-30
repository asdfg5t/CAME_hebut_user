"""PyCharm click-to-run adapters; experiment protocols remain in the tested modules."""
import argparse
import atexit
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from .author_loader import ROOT, fingerprint, verify_originals
from .filesystem_paths import windows_io_path
from .atomic_io import atomic_write_json, read_json_file

STAGES = ('preflight', 'pretrain_representation', 'evaluate_representation',
          'pretrain_few_shot', 'evaluate_few_shot', 'baselines', 'ablation', 'tensorboard')
CORPORA = {'representation': ['ogbn-products', 'books-lp', 'cloth-copurchase'],
           'few_shot': ['ogbn-products', 'books-nc', 'cloth-copurchase']}


def read_json(path):
    return read_json_file(path)


def write_json(path, value):
    atomic_write_json(path, value)


def validate_workflow(config):
    if config['schema'] != 'came_pycharm_v1':
        raise ValueError('Unknown PyCharm workflow schema')
    if type(config['pretraining_seed']) is not int or not 0 <= config['pretraining_seed'] < 2**32:
        raise ValueError('Pretraining seed must be an integer in the NumPy seed range')
    for name in ('evaluation_seeds', 'wiki_split_ids'):
        values = config[name]
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError('Nonempty distinct nonnegative integers required: ' + name)
    if any(v >= 2**32 for v in config['evaluation_seeds']):
        raise ValueError('Evaluation seed exceeds the NumPy seed range')
    if any(v >= 20 for v in config['wiki_split_ids']):
        raise ValueError('Wiki-CS has twenty official splits, numbered 0..19')
    if config['inference_batch_size'] < 1 or config['probe_batch_size'] < 0:
        raise ValueError('Invalid inference/probe batch size')


def completed_encoder(protocol, config, root=ROOT):
    """Select by successful completion and time; no test metric participates."""
    explicit = config[protocol + '_encoder']
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError('Configured complete encoder is missing: ' + str(path))
        return path.resolve()
    candidates = []
    for run in (root / 'runs').glob('*'):
        if not run.is_dir() or not all((run / name).is_file() for name in ('config.json', 'summary.json', 'status.json', 'manifest.json')):
            continue
        saved, summary, status, manifest = [read_json(run / name) for name in ('config.json', 'summary.json', 'status.json', 'manifest.json')]
        if (status['status'] == 'completed_pretraining' and summary.get('completed_epochs') == 8
                and summary.get('complete_declared_pretraining') is True and saved.get('datasets') == CORPORA[protocol]
                and saved.get('variant', 'came') == 'came' and saved.get('seed') == config['pretraining_seed']):
            path = Path(manifest['checkpoint_directory']) / 'encoder_final.pt'
            if path.is_file():
                candidates.append((manifest['created_utc'], str(run), path))
    if not candidates:
        raise RuntimeError('Finish the correct eight-epoch ' + protocol + ' pretraining first. Partial pilots are not accepted.')
    return max(candidates)[2].resolve()


def verify_complete_encoder(path, protocol):
    import torch
    from .joint_train import validate_joint
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_joint(state['config'])
    complete = state.get('completed_epochs', state.get('schedule', {}).get('epoch'))
    if (complete != 8 or state['config']['datasets'] != CORPORA[protocol]
            or state['config'].get('variant', 'came') != 'came'):
        raise ValueError('Evaluation requires the completed MAIN encoder of the correct corpus')


def evaluation_jobs(config, representation, few_shot=False, checkpoint=None):
    if few_shot:
        tasks = [('few_shot', dataset) for dataset in ('books-nc', 'ogbn-arxiv', 'wiki-cs')]
    else:
        tasks = [('node', dataset) for dataset in ('ele-fashion', 'wiki-cs', 'ogbn-arxiv', 'books-nc', 'ogbn-products')]
        tasks += [('link', dataset) for dataset in ('sports-copurchase', 'cloth-copurchase', 'books-lp')]
        tasks += [('relation', dataset) for dataset in ('FB15K237', 'WN18RR')]
    jobs = []
    for task, dataset in tasks:
        if representation == 'clip_image' and dataset not in ('ele-fashion', 'books-nc', 'sports-copurchase', 'cloth-copurchase', 'books-lp'):
            continue
        splits = config['wiki_split_ids'] if task == 'node' and dataset == 'wiki-cs' else [0]
        for split_id in splits:
            for seed in config['evaluation_seeds']:
                module = 'came_repro.link_evaluation' if task == 'link' else 'came_repro.downstream'
                arguments = ['-m', module, '--representation', representation, '--dataset', dataset, '--seed', str(seed)]
                if checkpoint:
                    arguments += ['--checkpoint', str(checkpoint)]
                if task != 'link':
                    arguments += ['--task', task, '--split-id', str(split_id), '--probe-batch-size', str(config['probe_batch_size'])]
                arguments += ['--inference-batch-size', str(config['inference_batch_size'])]
                jobs.append({'task': task, 'dataset': dataset, 'representation': representation,
                             'seed': seed, 'split_id': split_id, 'arguments': arguments})
    return jobs


def stage_jobs(stage, config, dry_run=False):
    if stage == 'preflight':
        return [{'arguments': [str(ROOT / 'scripts/preflight_reproduction.py')]}]
    if stage == 'tensorboard':
        return [{'arguments': ['-m', 'tensorboard.main', '--logdir', windows_io_path(ROOT / 'runs'), '--host', '127.0.0.1', '--port', '6006']}]
    if stage.startswith('pretrain_') or stage == 'ablation':
        protocol = stage.removeprefix('pretrain_')
        profile = ROOT / config['ablation_config'] if stage == 'ablation' else ROOT / 'configs' / ('joint_' + protocol + '_reconstruction.json')
        from .joint_train import validate_joint
        saved = read_json(profile)
        validate_joint(saved)
        if saved['training']['max_steps'] is not None or any(v is not None for v in saved['sampling']['seed_limits'].values()):
            raise ValueError('Click-to-run training must preserve complete declared budgets')
        arguments = ['-m', 'came_repro.joint_train', '--config', str(profile), '--seed', str(config['pretraining_seed'])]
        resume = config['ablation_resume_checkpoint'] if stage == 'ablation' else config[protocol + '_resume_checkpoint']
        if resume:
            resume = Path(resume)
            arguments += ['--resume', str(resume if resume.is_absolute() else ROOT / resume)]
        return [{'arguments': arguments}]
    if stage == 'baselines':
        return sum((evaluation_jobs(config, view, few) for view in ('clip_text', 'clip_image') for few in (False, True)), [])
    protocol = stage.removeprefix('evaluate_')
    checkpoint = '<completed_' + protocol + '_encoder>' if dry_run else completed_encoder(protocol, config)
    if not dry_run:
        verify_complete_encoder(checkpoint, protocol)
    return evaluation_jobs(config, 'came', protocol == 'few_shot', checkpoint)


def run_child(arguments, logger):
    process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=str(ROOT), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                               env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'})
    run_directory = None
    try:
        for line in process.stdout:
            line = line.rstrip('\r\n')
            logger.info('%s', line)
            if line.startswith('RUN_DIR='):
                run_directory = Path(line.removeprefix('RUN_DIR='))
        if process.wait() != 0:
            raise RuntimeError('Child stage failed; inspect its run logs and the workflow log')
        return run_directory
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
        process.stdout.close()


def acquire_gpu_lock(path=None):
    """OS lock is released on exit; a leftover lock file does not block a later run."""
    path = Path(path) if path else ROOT / '.runtime/pycharm_gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError('Another CAME PyCharm GPU stage is active. Finish or stop it first; TensorBoard can run alongside it.') from error
    atexit.register(handle.close)
    return handle


def main(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workflow-config', type=Path, default=ROOT / 'configs/pycharm_workflow.json')
    args = parser.parse_args()
    expected = ROOT / '.venv-came39/Scripts/python.exe'
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError('Set the PyCharm run interpreter to: ' + str(expected))
    if stage not in STAGES:
        raise ValueError('Unknown workflow stage')
    os.chdir(ROOT)
    config = read_json(args.workflow_config)
    validate_workflow(config)
    verify_originals()
    jobs = stage_jobs(stage, config, args.dry_run)
    if args.dry_run:
        print(json.dumps({'stage': stage, 'executable': sys.executable, 'cwd': str(ROOT),
                          'jobs': len(jobs), 'commands': jobs, 'training_started': False}, indent=2))
        return
    gpu_lock = acquire_gpu_lock() if stage != 'tensorboard' else None
    from .storage_budget import check_budget
    check_budget(additional_cache_bytes=2 * 2**30 if stage.startswith('pretrain_') or stage == 'ablation' else 0)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = ROOT / 'runs' / (stamp + '_pycharm_' + stage + '_' + uuid.uuid4().hex[:6])
    (directory / 'logs').mkdir(parents=True)
    logger = logging.getLogger(str(directory))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
    formatter.converter = __import__('time').gmtime
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / 'logs/workflow.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    write_json(directory / 'config.json', config)
    write_json(directory / 'plan.json', jobs)
    code = {str(path.relative_to(ROOT)): fingerprint(path) for path in (ROOT / 'src/came_repro').glob('*.py')}
    profiles = {str(ROOT / 'configs/storage.json'): fingerprint(ROOT / 'configs/storage.json')}
    for job in jobs:
        arguments = job['arguments']
        for flag in ('--config', '--resume'):
            if flag in arguments:
                path = Path(arguments[arguments.index(flag) + 1])
                profiles[str(path)] = fingerprint(path)
    data_manifests = {str(path.relative_to(ROOT)): fingerprint(path) for path in (ROOT / 'data/processed').glob('*/*/manifest.json')}
    identity = {'stage': stage, 'config': config, 'jobs': jobs, 'code': code,
                'profile_sha256': profiles, 'prepared_manifest_sha256': data_manifests,
                'encoder_sha256': fingerprint(completed_encoder(stage.removeprefix('evaluate_'), config)) if stage.startswith('evaluate_') else None}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    ledger_directory = ROOT / 'artifacts/workflow_state'
    ledger_directory.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_directory / (stage + '_' + key + '.json')
    ledger = read_json(ledger_path) if ledger_path.exists() else {'identity': identity, 'completed': {}}
    write_json(directory / 'manifest.json', {'identity': identity, 'ledger': str(ledger_path), 'executable': sys.executable,
                                           'original_files': verify_originals(), 'paper_reproduction_complete': False})
    for relative in code:
        target = directory / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for script in (ROOT / 'scripts').glob('run_*.py'):
        target = directory / 'source/scripts' / script.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, target)
    try:
        for index, job in enumerate(jobs):
            if any(fingerprint(ROOT / path) != expected_hash for path, expected_hash in code.items()):
                raise ValueError('Code changed during the batch; stop rather than mix protocols')
            if any(fingerprint(path) != expected_hash for path, expected_hash in profiles.items()):
                raise ValueError('Configuration/resume checkpoint changed during the batch')
            logger.info('Job %d/%d: %s', index + 1, len(jobs), job)
            if str(index) in ledger['completed'] and stage not in ('preflight', 'tensorboard'):
                previous = ledger['completed'][str(index)]
                if fingerprint(Path(previous['run_directory']) / 'summary.json') != previous['summary_sha256']:
                    raise ValueError('A completed child summary changed; do not silently skip it')
                for path, expected_hash in previous.get('artifacts_sha256', {}).items():
                    if fingerprint(path) != expected_hash:
                        raise ValueError('A completed child artifact changed; do not silently skip it')
                logger.info('Verified completed child retained: %s', previous['run_directory'])
                continue
            check_budget()
            run = run_child(job['arguments'], logger)
            if stage == 'preflight':
                shutil.copy2(ROOT / 'artifacts/audit/preflight_reproduction.json', directory / 'preflight_report.json')
            if run is not None:
                status = read_json(run / 'status.json')['status']
                expected_status = 'completed_pretraining' if stage.startswith('pretrain_') or stage == 'ablation' else 'completed_downstream_task'
                if status != expected_status:
                    raise ValueError('A partial child cannot complete the full stage')
                artifacts = list(run.glob('test*.npz')) + list(run.glob('episode*.npz')) + list((run / 'figures').glob('*.png'))
                if expected_status == 'completed_pretraining':
                    artifacts.append(Path(read_json(run / 'manifest.json')['checkpoint_directory']) / 'encoder_final.pt')
                ledger['completed'][str(index)] = {'run_directory': str(run), 'summary_sha256': fingerprint(run / 'summary.json'),
                                                   'artifacts_sha256': {str(path): fingerprint(path) for path in artifacts}}
                write_json(ledger_path, ledger)
            write_json(directory / 'status.json', {'status': 'running', 'completed_jobs': index + 1, 'total_jobs': len(jobs)})
        write_json(directory / 'status.json', {'status': 'completed_stage', 'jobs': len(jobs), 'paper_reproduction_complete': False})
        logger.info('STAGE_COMPLETE=%s', stage)
    except BaseException as error:
        logger.exception('Workflow interrupted/failed')
        write_json(directory / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'error': str(error)})
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        if gpu_lock is not None:
            gpu_lock.close()
