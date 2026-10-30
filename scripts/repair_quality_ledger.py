"""Audit migration after adding diagnostics; preserve the paused original suite."""
import copy
import hashlib
import json
from pathlib import Path
import sys
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.author_loader import ROOT, fingerprint, verify_originals
from came_repro.atomic_io import read_json_file, atomic_write_json
from came_repro.full_suite import validate_completed


def main():
    state=ROOT/'artifacts/workflow_state'
    pointer=read_json_file(state/'full_suite_current.json')
    path=Path(pointer['ledger'])
    old=read_json_file(path)
    if old['state'] not in ('paused','ready'):
        raise ValueError('Only migrate a safely paused suite')
    verify_originals()
    previous=old['identity']['sources']
    current={str(p.relative_to(ROOT)):fingerprint(p) for pattern in ('src/came_repro/*.py','scripts/*.py') for p in ROOT.glob(pattern)}
    if current==previous:
        print('No source migration needed')
        return
    added={str(Path(p)) for p in ('src/came_repro/quality_protocol.py','src/came_repro/quality_pipeline.py',
       'src/came_repro/quality_pretrain.py','src/came_repro/quality_probe.py','scripts/run_quality_pilot.py',
       'scripts/run_first_seed_tasks.py','scripts/request_quality_pilot_pause.py','scripts/repair_quality_ledger.py')}
    downstream=str(Path('src/came_repro/downstream.py'))
    if previous.get(downstream)!='eb4e15a0ef639b35a1d1af957af993dd596b292d97377a8228452e2564fd6b0c':
        raise ValueError('Unexpected old downstream source; this migration is not applicable')
    changes={name:{'before':previous.get(name),'after':current.get(name)} for name in previous.keys()|current.keys() if previous.get(name)!=current.get(name)}
    if any(name!=downstream and (name not in added or name in previous or name not in current) for name in changes):
        raise ValueError('Unreviewed scientific/source change')
    for relative,digest in old['identity']['data_manifests'].items():
        if fingerprint(ROOT/relative)!=digest:
            raise ValueError('Data changed; refuse migration')
    report_path=ROOT/'artifacts/audit/quality_pilot_acceptance.json'
    if not report_path.is_file() or read_json_file(report_path).get('relevant_tests_passed') is not True:
        raise ValueError('Diagnostic and legacy-resume regression acceptance required first')
    for record in old['records'].values():
        if record['state']=='completed':
            validate_completed(record)
    new=copy.deepcopy(old)
    new['identity']['sources']=current
    key=hashlib.sha256(json.dumps(new['identity'],sort_keys=True).encode()).hexdigest()[:24]
    audit={'previous_ledger':str(path),'previous_sha256':fingerprint(path),'changes':changes,
           'reason':'Validation-only diagnostics added; old downstream default remains test-once with original protocol',
           'model_data_or_completed_results_changed':False,'elapsed_budget_reset':False,
           'completed_jobs_retained':sum(r['state']=='completed' for r in old['records'].values())}
    new['quality_infrastructure_migration']=audit
    target=state/('full_suite_'+key+'.json')
    if target.exists():
        raise ValueError('Target ledger already exists; review before replacing anything')
    atomic_write_json(target,new)
    pause=state/('full_suite_'+key+'_pause.json')
    atomic_write_json(pause,{'reason':'quality_pilot_switch_original_suite_remains_paused'})
    atomic_write_json(state/'full_suite_current.json',{'identity_key':key,'ledger':str(target),'pause_file':str(pause)})
    atomic_write_json(ROOT/'artifacts/audit/quality_ledger_migration.json',audit)
    print('Paused original suite migrated; completed jobs and in-progress head checkpoint retained. No training launched.')


if __name__ == '__main__':
    main()
