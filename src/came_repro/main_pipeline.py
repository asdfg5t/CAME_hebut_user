"""One Run button for the two main CAME corpora and their downstream tasks."""
import argparse
import datetime
import logging
import os
from pathlib import Path
import sys
import uuid

from .author_loader import ROOT, fingerprint, verify_originals
from .ide_workflow import main as run_stage, read_json, write_json, stage_jobs, validate_workflow

MAIN_STAGES = ('preflight', 'pretrain_representation', 'evaluate_representation',
               'pretrain_few_shot', 'evaluate_few_shot')


def execute_stages(executor, before_stage=None, after_stage=None):
    """An exception propagates immediately; subsequent stages are never launched."""
    for index, stage in enumerate(MAIN_STAGES):
        if before_stage:
            before_stage(index, stage)
        executor(stage)
        if after_stage:
            after_stage(index, stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workflow-config', type=Path, default=ROOT / 'configs/pycharm_workflow.json')
    args = parser.parse_args()
    expected = ROOT / '.venv-came39/Scripts/python.exe'
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError('Set the PyCharm run interpreter to: ' + str(expected))
    config_path = args.workflow_config.resolve()
    config = read_json(config_path)
    validate_workflow(config)
    verify_originals()
    os.chdir(ROOT)
    if args.dry_run:
        import json
        print(json.dumps({'stages': [{'stage': stage, 'jobs': len(stage_jobs(stage, config, True))}
                                     for stage in MAIN_STAGES], 'training_started': False,
                          'full_pretraining_epochs_per_corpus': 8, 'cpu_fallback': False,
                          'comparison_experiments': 'CLIP baselines and ablations use the separate 06/07 entries.'}, indent=2))
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = ROOT / 'runs' / (stamp + '_main_pipeline_' + uuid.uuid4().hex[:6])
    (directory / 'logs').mkdir(parents=True)
    logger = logging.getLogger(str(directory))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
    formatter.converter = __import__('time').gmtime
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / 'logs/pipeline.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    config_hash = fingerprint(config_path)
    code = {str(path.relative_to(ROOT)): fingerprint(path) for path in (ROOT / 'src/came_repro').glob('*.py')}
    write_json(directory / 'config.json', config)
    write_json(directory / 'manifest.json', {'stages': MAIN_STAGES, 'workflow_config_sha256': config_hash,
                                            'code_sha256': code, 'original_files': verify_originals(),
                                            'full_training_started_by': 'User launching run_00_main_pipeline.py',
                                            'paper_reproduction_complete': False})
    def before_stage(index, stage):
        if fingerprint(config_path) != config_hash or any(fingerprint(ROOT / path) != value for path, value in code.items()):
            raise ValueError('Code or workflow configuration changed during the pipeline')
        logger.info('STAGE %d/%d: %s', index + 1, len(MAIN_STAGES), stage)
        write_json(directory / 'status.json', {'status': 'running', 'active_stage': stage, 'completed_stages': index})
    def after_stage(index, stage):
        logger.info('COMPLETED STAGE=%s', stage)
        write_json(directory / 'status.json', {'status': 'running', 'completed_stages': index + 1})
    previous_arguments = sys.argv
    sys.argv = [previous_arguments[0], '--workflow-config', str(config_path)]
    try:
        execute_stages(run_stage, before_stage, after_stage)
        write_json(directory / 'status.json', {'status': 'completed_main_pipeline', 'completed_stages': len(MAIN_STAGES),
                                               'paper_reproduction_complete': False})
        logger.info('MAIN_PIPELINE_COMPLETE; comparisons and author-detail reconciliation remain separate')
    except BaseException as error:
        logger.exception('Main pipeline stopped; subsequent stages were not launched')
        write_json(directory / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                               'error': str(error), 'paper_reproduction_complete': False})
        raise
    finally:
        sys.argv = previous_arguments
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == '__main__':
    main()
