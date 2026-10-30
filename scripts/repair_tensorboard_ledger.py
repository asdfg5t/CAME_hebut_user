"""One-time audited migration for the first-job TensorBoard initialization failure."""
import copy
import hashlib
import json
from pathlib import Path
import sys
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import dgl
from came_repro.author_loader import ROOT,fingerprint,verify_originals
from came_repro.full_suite import build_jobs
from came_repro.ide_workflow import read_json,write_json,acquire_gpu_lock


def main():
    suite_lock=acquire_gpu_lock(ROOT/'.runtime/full_suite.lock')
    gpu_lock=None
    try:
        gpu_lock=acquire_gpu_lock()
        pointer_path=ROOT/'artifacts/workflow_state/full_suite_current.json'
        pointer=read_json(pointer_path)
        previous_path=Path(pointer['ledger'])
        previous=read_json(previous_path)
        profile=read_json(ROOT/'configs/full_suite_resumable.json')
        jobs=build_jobs(profile,read_json(ROOT/profile['training_config']))
        sources={str(path.relative_to(ROOT)):fingerprint(path) for pattern in ('src/came_repro/*.py','scripts/*.py') for path in ROOT.glob(pattern)}
        identity={'profile':profile,'jobs':jobs,'sources':sources,
                  'data_manifests':{str(path.relative_to(ROOT)):fingerprint(path) for path in (ROOT/'data/processed').glob('*/*/manifest.json')},
                  'storage':read_json(ROOT/'configs/storage.json'),
                  'environment':{'executable':sys.executable,'python':sys.version,'torch':torch.__version__,'dgl':dgl.__version__,'cuda':torch.version.cuda},
                  'originals':verify_originals()}
        for key in identity:
            if key!='sources' and identity[key]!=previous['identity'][key]:
                raise ValueError('Refusing migration: scientific protocol/data/environment changed: '+key)
        old_sources=previous['identity']['sources']
        changed={name for name in sources.keys()|old_sources.keys() if sources.get(name)!=old_sources.get(name)}
        permitted={'src/came_repro/experiment.py','src/came_repro/filesystem_paths.py','src/came_repro/ide_workflow.py',
                   'src/came_repro/full_suite.py','scripts/run_full_suite.py','scripts/repair_tensorboard_ledger.py'}
        if not {name.replace('\\','/') for name in changed}<=permitted:
            raise ValueError('Refusing migration: source changes extend beyond the reviewed logging/exit-code repair')
        if len(previous['records'])!=1:
            raise ValueError('This one-time repair only applies before the first training job')
        record=previous['records'].get(jobs[0]['id'])
        if record is None or record['state']!='failed' or any(a.get('run_directory') for a in record['attempts']):
            raise ValueError('Refusing migration: training progress may already exist')
        failed=ROOT/'runs/20261011T103605Z_suite_pretrain_representation_came_seed42_seed42_2657d7'
        checkpoint_root=Path(read_json(failed/'checkpoints/location.json')['checkpoint_directory'])
        if list(checkpoint_root.glob('*.pt')) or (failed/'manifest.json').exists() or (failed/'config.json').exists():
            raise ValueError('Refusing migration: failed initialization already produced training state')
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
        target=previous_path.parent/('full_suite_'+key+'.json')
        if target.exists():
            raise ValueError('Refusing to overwrite an existing suite ledger')
        migrated=copy.deepcopy(previous)
        migrated['identity']=identity
        migrated['state']='ready_after_logging_repair'
        migrated['repair_migration']={'previous_ledger':str(previous_path),'previous_sha256':fingerprint(previous_path),
                   'reason':'TensorBoard bytes path exceeded MAX_PATH before any optimizer update',
                   'changed_sources':{name:{'before':old_sources.get(name),'after':sources.get(name)} for name in changed},
                   'scientific_protocol_changed':False,'elapsed_budget_reset':False}
        write_json(target,migrated)
        write_json(pointer_path,{'identity_key':key,'ledger':str(target),'pause_file':str(target.with_name('full_suite_'+key+'_pause.json'))})
        assert fingerprint(previous_path)==migrated['repair_migration']['previous_sha256']
        write_json(ROOT/'artifacts/audit/tensorboard_ledger_migration.json',{'previous_ledger':str(previous_path),'new_ledger':str(target),
                   'preserved_active_seconds':migrated['active_seconds'],'no_optimizer_checkpoints':True,
                   'migration':migrated['repair_migration'],'original_files':verify_originals()})
        print('LOGGING_REPAIR_READY: old ledger/history and cumulative runtime preserved; run scripts/run_full_suite.py again.',flush=True)
    finally:
        if gpu_lock:
            gpu_lock.close()
        suite_lock.close()


if __name__=='__main__':
    main()
