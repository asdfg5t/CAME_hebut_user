"""Audit a paused calibration-only identity migration; never launch training."""
import datetime
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from came_repro.author_loader import fingerprint,verify_originals
from came_repro.atomic_io import atomic_write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    verify_originals()
    products=ROOT/'artifacts/delivery/products_sampled_stream_release.json'
    remaining=ROOT/'artifacts/delivery/remaining_paper_tasks_release.json'
    ledger_path=ROOT/'artifacts/workflow_state/products_sampled_stream_current.json'
    followup='--followup' in sys.argv
    bounded='--bounded-features' in sys.argv
    archive=ROOT/('artifacts/delivery/archive/products_before_bounded_features_20261021T235656Z' if bounded else 'artifacts/delivery/archive/products_before_ram_followup_20261021T235209Z' if followup else 'artifacts/delivery/archive/products_before_ram_repair_20261021T234537Z')
    old=read(archive/'artifacts/delivery/products_sampled_stream_release.json')
    old_hash=fingerprint(archive/'artifacts/delivery/products_sampled_stream_release.json')
    ledger=read(ledger_path)
    if ledger['release_sha256']!=old_hash:raise ValueError('Migration requires the exact preserved calibration release')
    if ledger['state']!='paused' or set(ledger['records'])!={'calibration'}:raise ValueError('Do not migrate formal training/feature/test artifacts')
    for a in ledger['records']['calibration']['attempts']:
        if a['returncode']!=75 or read(Path(a['run_directory'])/'status.json')['status']!='paused':raise ValueError('Only documented safe calibration pauses may migrate')
    if (ROOT/'artifacts/remaining_paper_tasks/shared_jobs.json').exists():raise ValueError('Cannot alter dependencies of an already-started remaining grid')
    changed={str(ROOT/x) for x in ['src/came_repro/products_sampled_stream.py','scripts/run_24_products_sampled_stream.py','tests/test_products_sampled_stream.py']}
    if followup or bounded:changed.update(str(ROOT/x) for x in ['src/came_repro/products_memory.py','scripts/prepare_products_ram_repair.py'])
    if bounded:changed.add(str(ROOT/'src/came_repro/products_controller_storage.py'))
    for path,digest in old['input_sha256'].items():
        if path not in changed and fingerprint(path)!=digest:raise ValueError('Unrelated frozen input changed: '+path)
    for name in ['goodreads_official_rebuild','goodreads_four_rounds']:
        for path,digest in read(ROOT/'artifacts/delivery'/(name+'_release.json'))['input_sha256'].items():
            if fingerprint(path)!=digest:raise ValueError('Protected Goodreads input changed: '+path)
    report=ROOT/('artifacts/audit/products_bounded_features_cpu_20261022.xml' if bounded else 'artifacts/audit/products_ram_repair_followup_cpu_20261022.xml' if followup else 'artifacts/audit/products_ram_repair_cpu_20261022.xml')
    suites=list(ET.parse(report).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    if counts['tests']!=(20 if bounded else 18 if followup else 17) or counts['failures'] or counts['errors'] or counts['skipped']:raise ValueError('Actual repair CPU acceptance required')
    audit=ROOT/('artifacts/audit/products_bounded_features_identity_20261022.json' if bounded else 'artifacts/audit/products_ram_identity_migration_followup_20261022.json' if followup else 'artifacts/audit/products_ram_identity_migration_20261022.json')
    atomic_write_json(audit,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_release_sha256=old_hash,archive=str(archive),original_ledger_sha256=fingerprint(ledger_path),
        prior_active_seconds=ledger['active_seconds'],cpu_tests=counts,science_configuration_sha256=fingerprint(ROOT/'configs/products_sampled_stream_20261021.json'),
        changes=['Read-only int32 full CSC; bitexact sampled int64 miniblocks','Detailed unchanged RAM guard','Safe pause clean exit'],
        GPU_acceptance_pending=True,formal_training_or_test_started=False,prior_runs_preserved=True))
    mapping=dict(old['input_sha256'])
    inputs=[Path(x) for x in changed]+[ROOT/'src/came_repro/products_memory.py',Path(__file__).resolve(),
        ROOT/'docs/products_ram_repair_20261022.md',report,audit]
    if followup or bounded:inputs+=[ROOT/'src/came_repro/products_controller_storage.py',ROOT/'docs/products_ram_followup_20261022.md']
    if bounded:inputs+=[ROOT/'docs/products_bounded_features_20261022.md']
    for path in inputs:mapping[str(path)]=fingerprint(path)
    new=dict(old,input_sha256=mapping,RAM_repair=dict(parent_release_sha256=old_hash,audit=str(audit),CPU_tests=counts['tests'],actual_GPU_acceptance_pending=True))
    atomic_write_json(products,new)
    new_hash=fingerprint(products)
    ledger.update(release_sha256=new_hash,state='ready_for_repaired_calibration',actual_Products_GPU_started=True,
        actual_cuda_calibration_complete=False,waiting_for_user_to_launch=True)
    ledger['records']['calibration']['state']='paused'
    ledger.setdefault('identity_migrations',[]).append(dict(parent_release_sha256=old_hash,release_sha256=new_hash,audit=str(audit),
        preserve_active_seconds=ledger['active_seconds'],formal_artifacts_reused=False))
    atomic_write_json(ledger_path,ledger)
    previous=read(remaining);old_remaining=fingerprint(remaining)
    for path,digest in previous['input_sha256'].items():
        if path not in changed and fingerprint(path)!=digest:raise ValueError('Unexpected remaining-grid source mutation: '+path)
    previous['input_sha256'].update({str(path):fingerprint(path) for path in inputs})
    previous['RAM_dependency_migration']=dict(parent_release_sha256=old_remaining,products_release_sha256=new_hash,
        reason='Grid never started; Products structural memory dependency updated after a preserved calibration pause',audit=str(audit))
    atomic_write_json(remaining,previous)
    print(json.dumps(dict(products_release_sha256=new_hash,remaining_release_sha256=fingerprint(remaining),prior_active_seconds_preserved=ledger['active_seconds'],CPU_tests=counts['tests'])))


if __name__=='__main__':main()
