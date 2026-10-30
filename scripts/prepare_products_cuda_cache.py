"""Freeze one predeclared unlabeled resource/fidelity candidate after rejected calibration."""
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
    originals=verify_originals()
    archive=ROOT/'artifacts/delivery/archive/products_before_cuda_cache_20261022T034325Z'
    products=ROOT/'artifacts/delivery/products_sampled_stream_release.json'
    remaining=ROOT/'artifacts/delivery/remaining_paper_tasks_release.json'
    ledger_path=ROOT/'artifacts/workflow_state/products_sampled_stream_current.json'
    old=read(archive/'artifacts/delivery/products_sampled_stream_release.json')
    parent=fingerprint(archive/'artifacts/delivery/products_sampled_stream_release.json')
    ledger=read(ledger_path)
    if ledger['release_sha256']!=parent or ledger['state']!='calibration_rejected_no_full_workload_started':raise ValueError('Expected immutable rejected resource calibration')
    if set(ledger['records'])!={'calibration'} or ledger['records']['calibration']['summary']['passed'] is not False:
        raise ValueError('Never migrate formal features, heads or test into a new inference scheme')
    if (ROOT/'artifacts/remaining_paper_tasks/shared_jobs.json').exists():raise ValueError('Cannot refresh an already-started dependent grid')
    record=ledger['records']['calibration']
    for path,digest in record['artifacts'].items():
        if fingerprint(path)!=digest:raise ValueError('Rejected calibration evidence changed: '+path)
    if record['summary']['test_evaluations']!=0:raise ValueError('Test-scored calibration cannot choose inference')
    changed={str(ROOT/x) for x in ['src/came_repro/products_sampled_stream.py','src/came_repro/products_memory.py',
        'scripts/run_24_products_sampled_stream.py','tests/test_products_sampled_stream.py','configs/products_sampled_stream_20261021.json']}
    for path,digest in old['input_sha256'].items():
        if path not in changed and fingerprint(path)!=digest:raise ValueError('Unrelated dependency changed: '+path)
    previous=read(remaining);previous_remaining=fingerprint(remaining)
    for path,digest in previous['input_sha256'].items():
        if path not in changed and fingerprint(path)!=digest:raise ValueError('Unexpected dependent-grid mutation: '+path)
    for name in ['goodreads_official_rebuild','goodreads_four_rounds']:
        for path,digest in read(ROOT/'artifacts/delivery'/(name+'_release.json'))['input_sha256'].items():
            if fingerprint(path)!=digest:raise ValueError('Protected completed Goodreads source changed: '+path)
    report=ROOT/'artifacts/audit/products_cuda_cache_cpu_20261022.xml'
    suites=list(ET.parse(report).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','errors','failures','skipped']}
    if counts!={'tests':22,'errors':0,'failures':0,'skipped':0}:raise ValueError('Actual22 CPU checks required')
    plan=ROOT/'docs/products_cuda_cache_plan_20261022.md'
    audit=ROOT/'artifacts/audit/products_cuda_cache_identity_20261022.json'
    atomic_write_json(audit,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),originals=originals,
        parent_release_sha256=parent,archive=str(archive),parent_rejected_run=record['run_directory'],
        parent_summary=record['summary'],parent_summary_sha256=fingerprint(Path(record['run_directory'])/'summary.json'),
        prior_active_seconds=ledger['active_seconds'],cpu_tests=counts,predeclared_plan_sha256=fingerprint(plan),
        new_candidate_count=1,test_used_for_selection=False,CUDA_acceptance_pending=True,
        scientific_adaptation='fanouts15/15 batch128; exact4GiBFP32 structural row cache; official full splits retained'))
    mapping=dict(old['input_sha256'])
    inputs=[Path(x) for x in changed]+[Path(__file__).resolve(),plan,report,audit]
    for path in inputs:mapping[str(path)]=fingerprint(path)
    new=dict(old,input_sha256=mapping,unlabeled_input_cache_revision=dict(parent_release_sha256=parent,
        audit=str(audit),single_predeclared_candidate=True,CUDA_acceptance_pending=True,test_accuracy=None))
    atomic_write_json(products,new);new_hash=fingerprint(products)
    ledger.setdefault('prior_calibration_records',[]).append(dict(parent_release_sha256=parent,record=record))
    ledger.update(release_sha256=new_hash,state='ready_for_cached_unlabeled_calibration',records={'calibration':dict(state='ready',attempts=[])},
        actual_cuda_calibration_complete=False,test_accuracy=None,waiting_for_user_to_launch=True)
    ledger.setdefault('identity_migrations',[]).append(dict(parent_release_sha256=parent,release_sha256=new_hash,
        audit=str(audit),preserve_active_seconds=ledger['active_seconds'],official_train_or_test_reused=False))
    atomic_write_json(ledger_path,ledger)
    previous['input_sha256'].update({str(path):fingerprint(path) for path in inputs})
    previous['Products_unstarted_dependency_revision']=dict(parent_release_sha256=previous_remaining,
        products_release_sha256=new_hash,audit=str(audit),remaining_grid_ever_started=False)
    atomic_write_json(remaining,previous)
    print(json.dumps(dict(products_release_sha256=new_hash,remaining_release_sha256=fingerprint(remaining),
        preserved_active_seconds=ledger['active_seconds'],GPU_acceptance_pending=True)))


if __name__=='__main__':main()
