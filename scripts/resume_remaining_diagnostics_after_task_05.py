"""Honor user task05 priority, then resume only the already accepted single-seed repairs."""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    path=ROOT/'artifacts/workflow_state/after_task_05_diagnostics_current.json'
    pause=ROOT/'artifacts/workflow_state/convergence_repair_pause.json'
    sources={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    scripts={name:fingerprint(ROOT/name) for name in ['scripts/run_queued_convergence_repairs.py','scripts/run_convergence_candidate.py','scripts/run_products_single_validation.py',str(Path(__file__).resolve().relative_to(ROOT))]}
    profiles={name:fingerprint(ROOT/name) for name in ['configs/convergence_repair_20261014.json','configs/products_single_seed_20261014.json']}
    identity={'sources':sources,'scripts':scripts,'profiles':profiles}
    state={'state':'waiting_for_user_FB_repeat_completion','identity':identity,'formal_repeat_training_started_by_agent':False,'new_GPU_job_started':False}
    if path.exists():
        previous=read_json_file(path)
        if previous['identity']!=identity:raise ValueError('Queued diagnostic identity changed')
        if previous['state']=='completed_accepted_diagnostics':return
    atomic_write_json(path,state)
    guard=acquire_gpu_lock(ROOT/'.runtime/after_task_05_diagnostics.lock')
    try:
        while True:
            repeat=read_json_file(ROOT/'artifacts/workflow_state/validated_remaining_repeats_current.json')
            if repeat['state'] in ('failed','paused'):
                state['state']='deferred_FB_failure_or_user_pause';atomic_write_json(path,state);return
            if repeat['state']=='completed_ten_downstream_seeds':
                try:
                    lock=acquire_gpu_lock(ROOT/'.runtime/validated_remaining_repeats.lock');lock.close()
                    lock=acquire_gpu_lock();lock.close();break
                except RuntimeError:pass
            time.sleep(10)
        if repeat['statistics']['completed']!=10 or not repeat['statistics']['complete_repeat_grid']:
            raise ValueError('Complete FB repeat statistics required')
        verify_originals()
        for name,digest in {**sources,**scripts,**profiles}.items():
            if fingerprint(ROOT/name)!=digest:raise ValueError('Frozen diagnostic source/profile changed')
        ledger_path=ROOT/'artifacts/workflow_state/convergence_repair_current.json';ledger=read_json_file(ledger_path)
        for name,digest in ledger['identity']['sources'].items():
            if fingerprint(ROOT/name)!=digest:raise ValueError('Paused diagnostic source changed')
        for name,digest in ledger['identity']['scripts'].items():
            if fingerprint(ROOT/name)!=digest:raise ValueError('Paused diagnostic script changed')
        if fingerprint(ROOT/'configs/convergence_repair_20261014.json')!=ledger['identity']['profile_sha256']:
            raise ValueError('Paused convergence protocol changed')
        if ledger['state']=='paused':
            if not pause.exists() or read_json_file(pause).get('reason')!='user_priority_validated_FB15K237_repeats':
                state['state']='deferred_different_user_pause';atomic_write_json(path,state);return
            handoff=read_json_file(ROOT/'artifacts/audit/job_05_handoff_safe_pause_20261014.json')
            cp=Path(handoff['checkpoint'])
            if fingerprint(cp)!=handoff['checkpoint_sha256']:raise ValueError('Paused full checkpoint changed')
            saved=torch.load(str(cp),map_location='cpu',weights_only=False)
            if saved['schema']!='came_wolfe_linear_v1' or saved['completed_epochs']!=138 or saved['next_epoch']!=139 or len(saved['history_rows'])!=138:
                raise ValueError('Paused full solver boundary changed')
            if not saved['optimizer'] or not saved['rng']:raise ValueError('Complete optimizer/RNG required')
            auditdir=ROOT/'artifacts/audit/task05_resume';auditdir.mkdir(parents=True,exist_ok=True)
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            archive=auditdir/(stamp+'_priority_pause.json');archive.write_bytes(pause.read_bytes())
            atomic_write_json(auditdir/(stamp+'_resume_audit.json'),{'authorizing_instruction':'user requests remaining failed-task validated versions after task05','FB_complete_results':10,'FB_mean_accuracy_pct':repeat['statistics']['mean_accuracy_pct'],'paused_checkpoint':str(cp),'parent_sha256':fingerprint(cp),'schema':saved['schema'],'next_step':139,'paused_ledger_sha256':fingerprint(ledger_path),'pause_archive_sha256':fingerprint(archive),'source_profile_identity_unchanged':True,'formal_FB_repeats_not_restarted':True})
            # Non-recursive deletion of exactly the archived priority pause, within workspace.
            if pause.resolve().parent!=(ROOT/'artifacts/workflow_state').resolve():raise ValueError('Pause path escaped workspace')
            pause.unlink()
        state.update(state='resuming_accepted_convergence_diagnostics',new_GPU_job_started=True);atomic_write_json(path,state)
        if ledger['state']!='completed_bounded_convergence_screen':
            code=subprocess.run([sys.executable,'-u','scripts/run_queued_convergence_repairs.py'],cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8'}).returncode
            current=read_json_file(ledger_path)
            if code or current['state']!='completed_bounded_convergence_screen':
                state.update(state='deferred_convergence_pause' if current['state']=='paused' else 'failed_convergence',returncode=code);atomic_write_json(path,state);return
        state['state']='running_full_products_single_seed';atomic_write_json(path,state)
        product_path=ROOT/'artifacts/workflow_state/products_single_seed_current.json'
        if product_path.exists() and read_json_file(product_path)['state'] not in ('completed_verified_official_single_seed',):
            state['state']='deferred_existing_products_state_requires_resume_audit';atomic_write_json(path,state);return
        code=subprocess.run([sys.executable,'-u','scripts/run_products_single_validation.py'],cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8'}).returncode
        product=read_json_file(product_path)
        if code or product['state']!='completed_verified_official_single_seed':
            state.update(state='deferred_products_pause' if product['state']=='paused' else 'failed_products',returncode=code);atomic_write_json(path,state);return
        state.update(state='completed_accepted_diagnostics',all_unmet_passed_not_assumed=True);atomic_write_json(path,state)
    except BaseException as error:
        state.update(state='failed_resume_audit',error=str(error));atomic_write_json(path,state);raise
    finally:guard.close()


if __name__=='__main__':main()
