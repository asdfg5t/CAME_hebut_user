"""Reviewed batch-only resume migration; keep every historical run and prefix."""
import sys
from pathlib import Path
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import copy
import json
import hashlib
from came_repro.author_loader import ROOT, fingerprint, verify_originals
from came_repro.atomic_io import read_json_file,atomic_write_json


def main():
    state=ROOT/'artifacts/workflow_state'
    calibration=read_json_file(ROOT/'artifacts/audit/books_inference_batch_calibration_128.json')
    if not calibration['memory_calibration_passed'] or not calibration['output_equivalence_passed']:
        raise ValueError('Actual CUDA calibration required')
    if max(r['peak_allocated_mib'] for r in calibration['samples'])>4800:
        raise ValueError('Insufficient VRAM reserve')
    current={str(p.relative_to(ROOT)):fingerprint(p) for pattern in ('src/came_repro/*.py','scripts/*.py') for p in ROOT.glob(pattern)}
    allowed={str(Path(p)) for p in ('src/came_repro/fast_graph_screen.py','scripts/run_fast_graph_screen.py',
              'scripts/request_fast_graph_pause.py','scripts/calibrate_books_inference.py','scripts/repair_fast_graph_resume.py',
              'scripts/run_remaining_single_seed.py')}
    verify_originals()
    audit=[]
    for prefix in ('full_suite','quality_pilot','fast_graph_screen'):
        pointer_path=state/(prefix+'_current.json');pointer=read_json_file(pointer_path);path=Path(pointer['ledger']);old=read_json_file(path)
        if old['state']!='paused':raise ValueError('Only migrate paused ledgers: '+prefix)
        before=old['identity']['sources'];changes={n for n in before.keys()|current.keys() if before.get(n)!=current.get(n)}
        if any(n not in allowed or n in before or n not in current for n in changes):
            raise ValueError('Unexpected changed/deleted source: '+prefix)
        for record in old['records'].values():
            if record['state']=='completed':
                for p,digest in record['artifact_sha256'].items():
                    if fingerprint(p)!=digest:raise ValueError('Historical artifact changed')
        data_key='data_manifests' if prefix=='full_suite' else 'data'
        for relative,digest in old['identity'][data_key].items():
            if fingerprint(ROOT/relative)!=digest:raise ValueError('Data changed')
        new=copy.deepcopy(old);new['identity']['sources']=current
        retained_cursor=None
        if prefix=='fast_graph_screen':
            if old['identity']['encoder_sha256']!=calibration['encoder_sha256']:raise ValueError('Calibration model mismatch')
            profile=read_json_file(ROOT/'configs/fast_graph_screen.json');expected=copy.deepcopy(old['identity']['profile'])
            expected['inference_batches']['books-nc']=128
            if profile!=expected:raise ValueError('Only the calibrated inference batch may change')
            run=Path(old['records']['validate_books-nc']['attempts'][-1]['run_directory'])
            checkpoint=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])
            if (checkpoint/'probe_last.pt').exists():raise ValueError('Head already started; batch-only migration is not applicable')
            cache=Path(read_json_file(ROOT/'configs/storage.json')['embedding_cache_root'])/'quality_pilot'/calibration['encoder_sha256'][:20]/'books-nc_bidirected'
            saved=read_json_file(cache/'inference_state.json');retained_cursor=saved['cursor']
            if retained_cursor<=0 or saved['identity']['checkpoint_sha256']!=calibration['encoder_sha256']:
                raise ValueError('Missing or unrelated inference prefix')
            new['identity']['profile']=profile
        key=hashlib.sha256(json.dumps(new['identity'],sort_keys=True).encode()).hexdigest()[:24]
        target=state/(prefix+'_'+key+'.json')
        if target.exists():raise ValueError('Target ledger already exists')
        entry={'previous_ledger':str(path),'previous_sha256':fingerprint(path),'new_ledger':str(target),
               'source_additions':sorted(changes),'retained_inference_cursor':retained_cursor,
               'model_data_results_changed':False,'runtime_reset':False,
               'batch_change':'books-nc exact inference 8 -> 128, calibrated; no head/model change' if prefix=='fast_graph_screen' else None}
        new['fast_batch_migration']=entry;atomic_write_json(target,new)
        updated={**pointer,'identity_key':key,'ledger':str(target)}
        # Keep the pause pathname so an already displayed pause launcher remains valid.
        if prefix!='fast_graph_screen':atomic_write_json(Path(pointer['pause_file']),{'reason':'kept_paused_after_fast_additions'})
        atomic_write_json(pointer_path,updated);audit.append(entry)
    atomic_write_json(ROOT/'artifacts/audit/fast_graph_batch_resume_migration.json',audit)
    print('Batch-only migration reviewed; all completed results and the Books embedding prefix retained.')


if __name__=='__main__':
    main()
