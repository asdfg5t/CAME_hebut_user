"""Audited I/O-only migration that retains completed encoders and inference progress."""
import copy
import hashlib
import json
from pathlib import Path
import sys
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import dgl
import numpy as np
from came_repro.author_loader import ROOT, fingerprint, verify_originals
from came_repro.full_suite import build_jobs, validate_completed
from came_repro.ide_workflow import read_json, write_json, acquire_gpu_lock


def main():
    suite_lock = acquire_gpu_lock(ROOT / '.runtime/full_suite.lock')
    gpu_lock = None
    try:
        gpu_lock = acquire_gpu_lock()
        pointer_path = ROOT / 'artifacts/workflow_state/full_suite_current.json'
        pointer = read_json(pointer_path)
        previous_path = Path(pointer['ledger'])
        previous = read_json(previous_path)
        if previous['state'] != 'failed':
            raise ValueError('This repair applies to a stopped failed suite only')
        profile = read_json(ROOT / 'configs/full_suite_resumable.json')
        jobs = build_jobs(profile, read_json(ROOT / profile['training_config']))
        sources = {str(path.relative_to(ROOT)):fingerprint(path) for pattern in ('src/came_repro/*.py','scripts/*.py') for path in ROOT.glob(pattern)}
        identity = {'profile':profile,'jobs':jobs,'sources':sources,
                    'data_manifests':{str(path.relative_to(ROOT)):fingerprint(path) for path in (ROOT / 'data/processed').glob('*/*/manifest.json')},
                    'storage':read_json(ROOT / 'configs/storage.json'),
                    'environment':{'executable':sys.executable,'python':sys.version,'torch':torch.__version__,'dgl':dgl.__version__,'cuda':torch.version.cuda},
                    'originals':verify_originals()}
        for name in identity:
            if name != 'sources' and identity[name] != previous['identity'][name]:
                raise ValueError('Scientific protocol/data/environment changed: ' + name)
        old_sources = previous['identity']['sources']
        changed = {name for name in sources.keys() | old_sources.keys() if sources.get(name) != old_sources.get(name)}
        permitted = {'src/came_repro/atomic_io.py','src/came_repro/experiment.py','src/came_repro/ide_workflow.py',
                     'src/came_repro/stream_inference.py','scripts/request_full_suite_pause.py','scripts/repair_atomic_io_ledger.py'}
        if not {name.replace('\\','/') for name in changed} <= permitted:
            raise ValueError('Unreviewed source changes outside the I/O repair')
        completed = [record for record in previous['records'].values() if record['state'] == 'completed']
        for record in completed:
            validate_completed(record)
        main = previous['records']['pretrain_representation_came_seed42']
        if main['state'] != 'completed' or main['summary']['completed_subset_rounds'] != 8:
            raise ValueError('Completed eight-round encoder must be preserved')
        encoder_hash = fingerprint(main['encoder_checkpoint'])
        cache = Path(identity['storage']['embedding_cache_root']) / encoder_hash[:20] / 'ele-fashion'
        state = read_json(cache / 'inference_state.json')
        if state['identity']['checkpoint_sha256'] != encoder_hash or state['complete'] or state['cursor'] != 24960:
            raise ValueError('Unexpected failed inference cursor/model; investigate before migration')
        embeddings = np.load(cache / 'fused_embeddings.npy', mmap_mode='r')
        if embeddings.shape != (97766, 1024) or embeddings.dtype != np.float32:
            raise ValueError('Saved inference array has an unexpected shape/type')
        for start in range(0, state['cursor'], 512):
            if not np.isfinite(embeddings[start:min(state['cursor'],start+512)]).all():
                raise ValueError('Nonfinite committed inference rows')
        del embeddings
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        target = previous_path.parent / ('full_suite_' + key + '.json')
        if target.exists():
            raise ValueError('Refusing to overwrite a suite ledger')
        migrated = copy.deepcopy(previous)
        migrated['identity'] = identity
        migrated['state'] = 'ready_after_atomic_io_repair'
        migration = {'previous_ledger':str(previous_path),'previous_sha256':fingerprint(previous_path),
                     'reason':'Bounded Windows atomic file replacement retry; model/math/data unchanged',
                     'changed_sources':{name:{'before':old_sources.get(name),'after':sources.get(name)} for name in changed},
                     'completed_jobs_retained':len(completed),'encoder_sha256':encoder_hash,'inference_cursor_retained':state['cursor'],
                     'elapsed_budget_reset':False,'scientific_protocol_changed':False}
        migrated['atomic_io_repair_migration'] = migration
        write_json(target, migrated)
        write_json(pointer_path, {'identity_key':key,'ledger':str(target),'pause_file':str(target.with_name('full_suite_' + key + '_pause.json'))})
        assert fingerprint(previous_path) == migration['previous_sha256']
        write_json(ROOT / 'artifacts/audit/atomic_io_ledger_migration.json', {'migration':migration,'new_ledger':str(target),
                   'preserved_active_seconds':migrated['active_seconds'],'saved_embedding_prefix_finite':True,'original_files':verify_originals()})
        print('ATOMIC_IO_REPAIR_READY: completed encoder, 24960 inferred nodes, old history and cumulative time retained.', flush=True)
    finally:
        if gpu_lock:
            gpu_lock.close()
        suite_lock.close()


if __name__ == '__main__':
    main()
