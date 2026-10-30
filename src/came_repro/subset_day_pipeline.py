"""Quality-first, wall-clock-bounded subset reproduction; never promise metric parity."""
import argparse
import copy
import csv
import datetime
import json
import logging
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

from .author_loader import ROOT, fingerprint, verify_originals
from .budgeted_pipeline import run_bounded, result_row
from .ide_workflow import acquire_gpu_lock, read_json, write_json


def validate_plan(plan, training):
    from .joint_train import validate_joint
    validate_joint(training)
    if not 0 < plan['total_wall_hours'] <= 24 or training['training']['max_steps'] is not None:
        raise ValueError('At most 24 hours; eight complete subset rounds requested without a step cap')
    if plan['probe_max_epochs'] != 2000 or plan['primary_dataset'] != 'ele-fashion':
        raise ValueError('This quality profile preserves the paper probe limit on Ele-fashion')
    if plan['evaluation_seeds'] != [0, 1, 2]:
        raise ValueError('Three explicit downstream initializations required')
    if any(not isinstance(v, int) or v < 1 for v in training['sampling']['seed_limits'].values()):
        raise ValueError('Fixed seed pools must be explicit positive counts')
    for name in ('main_wall_hours', 'ablation_wall_hours', 'probe_wall_minutes', 'probe_patience', 'inference_batch_size'):
        if plan[name] <= 0:
            raise ValueError('Positive resource budget required: ' + name)


def update_budget(training):
    batch = training['sampling']['seed_batch_size']
    batches = sum((count + batch - 1) // batch for count in training['sampling']['seed_limits'].values())
    return {'fixed_seed_pools': training['sampling']['seed_limits'], 'batches_per_subset_round': batches,
            'subset_rounds': training['training']['epochs'], 'planned_batches': batches * training['training']['epochs'],
            'full_graph_epoch_claim': False, 'author_subset_size_known': False}


def probe_arguments(plan, seed, representation, checkpoint=None):
    arguments = ['-m', 'came_repro.downstream', '--dataset', plan['primary_dataset'], '--task', 'node',
                 '--representation', representation, '--seed', str(seed), '--epochs', '2000',
                 '--patience', str(plan['probe_patience']), '--inference-batch-size', str(plan['inference_batch_size'])]
    if checkpoint:
        arguments += ['--checkpoint', str(checkpoint)]
    return arguments


def own_checkpoint(run_directory):
    if not run_directory:
        return None
    directory = Path(run_directory)
    if not (directory / 'manifest.json').is_file():
        return None
    root = Path(read_json(directory / 'manifest.json')['checkpoint_directory'])
    for name in ('encoder_final.pt', 'last.pt'):
        if (root / name).is_file():
            return root / name
    return None


def aggregate_results(records):
    groups = {}
    for row in records:
        if row.get('task') != 'probe' or row['status'] != 'completed':
            continue
        accuracy = row.get('summary', {}).get('test_accuracy')
        if accuracy is None:
            continue
        groups.setdefault(row['variant'], []).append(100 * accuracy)
    return [{ 'variant': name, 'completed_downstream_seeds': len(values),
              'mean_accuracy_percent': statistics.mean(values),
              'sample_std_percent': statistics.stdev(values) if len(values) > 1 else None,
              'raw_accuracy_percent': values} for name, values in groups.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/subset_day_pipeline.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = read_json(args.profile)
    training = read_json(ROOT / plan['training_config'])
    validate_plan(plan, training)
    verify_originals()
    if Path(sys.executable).resolve() != (ROOT / '.venv-came39/Scripts/python.exe').resolve():
        raise RuntimeError('Verified project CUDA environment required')
    if args.dry_run:
        print(json.dumps({'plan': plan, 'training_budget': update_budget(training), 'training_started': False,
                          'duration_and_metric_gap_guaranteed': False}, ensure_ascii=False, indent=2))
        return
    lock = acquire_gpu_lock()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = ROOT / 'runs' / (stamp + '_' + plan['name'] + '_' + uuid.uuid4().hex[:6])
    (directory / 'logs').mkdir(parents=True)
    logger = logging.getLogger(str(directory))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
    formatter.converter = time.gmtime
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / 'logs/pipeline.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    started = time.monotonic()
    deadline = started + plan['total_wall_hours'] * 3600
    records, sources = [], {}
    write_json(directory / 'config.json', plan)
    for pattern in ('src/came_repro/*.py', 'scripts/*.py', 'configs/**/*.json'):
        for path in ROOT.glob(pattern):
            target = directory / 'source' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            sources[str(path.relative_to(ROOT))] = fingerprint(path)
    write_json(directory / 'manifest.json', {'originals': verify_originals(), 'sources_sha256': sources,
               'executable': sys.executable, 'pretraining_seed': training['seed'], 'scope': plan['scope'],
               'paper_reproduction_complete': False, 'training_budget': update_budget(training)})
    generated = directory / 'resolved_training'
    generated.mkdir()

    def remaining():
        return max(0., deadline - time.monotonic())

    def save():
        aggregate = aggregate_results(records)
        write_json(directory / 'summary.json', {'records': records, 'aggregate': aggregate,
                   'elapsed_hours': (time.monotonic() - started) / 3600, 'paper_reproduction_complete': False,
                   'scope': plan['scope'], 'time_limit_is_not_completion_guarantee': True})
        with (directory / 'results.csv').open('w', encoding='utf-8', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=['variant', 'seed', 'status', 'run_directory', 'test_accuracy_percent'])
            writer.writeheader()
            for row in records:
                if row.get('task') == 'probe':
                    accuracy = row.get('summary', {}).get('test_accuracy')
                    writer.writerow({key: row.get(key, '') for key in ('variant', 'seed', 'status', 'run_directory')} |
                                    {'test_accuracy_percent': 100 * accuracy if accuracy is not None and row['status'] == 'completed' else ''})
        write_json(directory / 'aggregate_results.json', aggregate)

    def launch(arguments, seconds, job):
        if remaining() <= 0:
            row = {**job, 'status': 'deferred', 'run_directory': None, 'reason': '24-hour wall budget exhausted'}
        else:
            if any(fingerprint(ROOT / path) != value for path, value in sources.items()):
                raise RuntimeError('Project source/configuration changed during experiment')
            logger.info('STAGE %s remaining_hours=%.2f', job, remaining() / 3600)
            outcome = run_bounded(arguments, min(seconds, remaining()), logger)
            row = result_row(job, outcome)
        records.append(row)
        save()
        return row

    def evaluate(variant, representation, checkpoint=None):
        for seed in plan['evaluation_seeds']:
            launch(probe_arguments(plan, seed, representation, checkpoint), plan['probe_wall_minutes'] * 60,
                   {'task': 'probe', 'variant': variant, 'seed': seed})

    def pretrain(variant, hours):
        config = copy.deepcopy(training)
        config['name'] = training['name'] + '_' + variant
        if variant != 'came':
            from .variants import expected_loss_weights
            config['variant'] = variant
            config['model'].update(expected_loss_weights(config['model'], variant))
            config['deviations'].append('Conditional ablation: same fixed pools, pretraining seed, architecture budget and eight subset rounds; variant=' + variant)
        path = generated / (variant + '.json')
        write_json(path, config)
        row = launch(['-m', 'came_repro.subset_offline_train', '--config', str(path)], hours * 3600,
                     {'task': 'pretraining', 'variant': variant, 'seed': config['seed']})
        checkpoint = own_checkpoint(row.get('run_directory'))
        if checkpoint:
            row['checkpoint'] = str(checkpoint)
            row['checkpoint_sha256'] = fingerprint(checkpoint)
            save()
        return row, checkpoint

    try:
        from .storage_budget import check_budget
        check_budget(additional_cache_bytes=2 * 2**30)
        logger.warning('Paper-supported Products subset/offline PPR; pool counts and other graph caps are adaptations. No metric/time-to-completion guarantee.')
        preflight = launch([str(ROOT / 'scripts/preflight_reproduction.py')], 1800, {'task': 'preflight'})
        if preflight['status'] != 'completed':
            raise RuntimeError('CUDA/data/storage preflight failed or timed out')
        main_row, checkpoint = pretrain('came', plan['main_wall_hours'])
        # Preserve a useful partial result but identify its exact completed subset rounds.
        if checkpoint:
            evaluate('came', 'came', checkpoint)
        else:
            records.append({'task': 'probe', 'variant': 'came', 'status': 'deferred', 'reason': 'No valid saved encoder'})
        for representation in ('clip_text', 'clip_image'):
            evaluate(representation, representation)
        for variant in plan['ablation_variants']:
            if not main_row.get('summary', {}).get('complete_declared_subset_pretraining'):
                records.append({'task': 'pretraining', 'variant': variant, 'status': 'deferred',
                                'reason': 'Main eight-round budget incomplete; avoid unequal-budget ablation comparison'})
                continue
            row, variant_checkpoint = pretrain(variant, plan['ablation_wall_hours'])
            if variant_checkpoint and row.get('summary', {}).get('complete_declared_subset_pretraining'):
                evaluate(variant, 'came', variant_checkpoint)
        save()
        if aggregate_results(records):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            values = aggregate_results(records)
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.bar([row['variant'] for row in values], [row['mean_accuracy_percent'] for row in values],
                     yerr=[row['sample_std_percent'] or 0 for row in values], capsize=4)
            axis.axhline(plan['paper_reference_accuracy_percent'], color='gray', linestyle='--', label='Paper CAME 85.13%; different budget/repeats')
            axis.set(ylabel='Official test accuracy (%)', ylim=(0, 100), title='Ele-fashion: validation-selected frozen probes')
            axis.tick_params(axis='x', labelrotation=15)
            axis.legend(fontsize=8)
            figure.text(.02, .01, 'Source: aggregate_results.json; actual completed seeds only; one pretraining seed; reduced seed pools. Error bars: sample SD.', fontsize=7)
            figure.tight_layout(rect=(0, .06, 1, 1))
            figure.savefig(directory / 'results_accuracy.png', dpi=180)
            plt.close(figure)
        write_json(directory / 'omitted_scope.json', {'omitted': ['Other nine downstream domains', 'Separate few-shot pretraining corpus',
                   'Seven other ablations', 'Seven extra downstream seeds and repeated pretraining seeds'], 'raw_data_deleted': False})
        write_json(directory / 'status.json', {'status': 'finished_with_recorded_outcomes', 'paper_reproduction_complete': False})
        print('DAY_RUN_DIR=' + str(directory), flush=True)
    except BaseException as error:
        logger.exception('Subset day pipeline stopped')
        save()
        write_json(directory / 'status.json', {'status': 'failed_or_interrupted', 'error': str(error)})
        raise
    finally:
        lock.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == '__main__':
    main()
