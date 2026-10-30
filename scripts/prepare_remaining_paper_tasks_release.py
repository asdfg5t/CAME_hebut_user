"""CPU-only release construction/verification; never starts scientific GPU work."""
import datetime
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from came_repro.atomic_io import atomic_write_json
from came_repro.author_loader import fingerprint,verify_originals
from came_repro.paper_task_groups import GROUPS,PROFILE,RELEASE,counts,jobs_for_group,read,reference_encoder


def main():
    originals=verify_originals()
    refresh='--refresh-unstarted' in sys.argv
    parent_release=None
    if refresh:
        if (ROOT/'artifacts/remaining_paper_tasks/shared_jobs.json').exists():raise RuntimeError('Cannot refresh an active or previously started grid')
        parent_release=fingerprint(RELEASE)
    if RELEASE.exists() and not refresh:
        for path,digest in read(RELEASE)['input_sha256'].items():
            if fingerprint(path)!=digest:raise ValueError('Release input changed: '+path)
        print(json.dumps(dict(state='existing_release_verified',release_sha256=fingerprint(RELEASE),originals=originals)))
        return
    dependencies={}
    protected=[]
    for name in ['goodreads_official_rebuild','goodreads_four_rounds','products_sampled_stream']:
        path=ROOT/'artifacts/delivery'/(name+'_release.json')
        mapping=read(path)['input_sha256']
        for file,digest in mapping.items():
            if fingerprint(file)!=digest:raise ValueError('Protected active dependency changed: '+file)
        protected.append(dict(path=str(path),sha256=fingerprint(path),verified_files=len(mapping)))
        dependencies.update(mapping)
    report=ROOT/'artifacts/audit/remaining_paper_tasks_optimized_cpu_20261021.xml'
    suites=list(ET.parse(report).getroot().iter('testsuite'))
    test_counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','errors','failures','skipped']}
    if test_counts['tests']<32 or test_counts['errors'] or test_counts['failures'] or test_counts['skipped']:
        raise ValueError('Complete relevant CPU checks required before freeze')
    historical=ROOT/'runs/20261011T105355Z_suite_pretra_0c927d_s42_9521d7'
    timestamps=re.findall(r'^([0-9-]{10} [0-9:,]{12}) ',(historical/'logs/train.log').read_text(encoding='utf-8'),flags=re.M)
    first,last=[datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S,%f') for x in [timestamps[0],timestamps[-1]]]
    historical_summary=read(historical/'summary.json')
    control={};reference_encoder(jobs_for_group('ablation_reference')[0],control)
    unique={j['id']:j for group in GROUPS for j in jobs_for_group(group)}
    audit=ROOT/'artifacts/audit/remaining_paper_tasks_optimized_preparation_20261021.json'
    atomic_write_json(audit,dict(stage='code_preparation_only_actual_gpu_pending',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cpu_tests=test_counts,parent_unstarted_release_sha256=parent_release,
        io_optimizations=['TensorBoard5s explicit flush; every CSV/JSONL row still immediate',
            'Reuse complete selected-head validation replay only after SHA, official IDs, head selection and MRR verification'],
        end_to_end_speedup_measured=False,ablation_pretraining_rounds_unchanged=True,
        actual_new_gpu_validation=False,new_training_started=False,original_sha256=originals,
        protected_releases=protected,group_counts={g:counts(jobs_for_group(g)) for g in GROUPS},
        unique_executable_jobs=sum(j['kind']!='reference_encoder' for j in unique.values()),matched_control=control,
        historical_timing=dict(run_directory=str(historical),log_sha256=fingerprint(historical/'logs/train.log'),
            summary_sha256=fingerprint(historical/'summary.json'),first_log_utc=str(first),last_log_utc=str(last),
            log_span_seconds=(last-first).total_seconds(),updates=historical_summary['step'],
            subset_rounds=historical_summary['completed_subset_rounds'],not_new_runtime_measurement=True),
        paper_pages=[6,7,8,14,15,16],unreleased_details=read(PROFILE)['unreleased_details'],
        new_group_checkpoint_sizes_and_GPU_timings_are_estimates=True,capacity_guard_preserves_existing_limits=True))
    inputs=[PROFILE,audit,report,ROOT/'configs/full_suite_resumable.json',ROOT/'configs/subset_offline_24h.json',
        ROOT/'docs/remaining_paper_tasks_20261021.md',ROOT/'docs/remaining_paper_tasks_quickstart_20261021.md',
        ROOT/'src/came_repro/paper_task_groups.py',ROOT/'src/came_repro/paper_task_runtime.py',ROOT/'src/came_repro/paper_task_io.py',
        ROOT/'tests/test_paper_task_groups.py',Path(__file__).resolve(),
        ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json']
    for n in range(25,38):
        matches=list((ROOT/'scripts').glob('run_%02d_*.py'%n))
        if len(matches)!=1:raise ValueError('Exactly one independent entry required: '+str(n))
        inputs+=matches
    for path in inputs:dependencies[str(path)]=fingerprint(path)
    atomic_write_json(RELEASE,dict(stage='independent_remaining_paper_tasks_code_ready_gpu_pending',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_sha256=dependencies,
        cpu_checks=test_counts,actual_new_GPU_acceptance=False,test_accuracy_not_guaranteed=True,
        existing_experiment_sources_changed=False,preparation_audit=str(audit)))
    print(json.dumps(dict(release_sha256=fingerprint(RELEASE),frozen_inputs=len(dependencies),cpu_tests=test_counts,
        unique_executable_jobs=1020,actual_new_GPU_acceptance=False),ensure_ascii=False))


if __name__=='__main__':main()
