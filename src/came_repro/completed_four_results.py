"""Read the immutable completed four-task report after new stage infrastructure is added."""
from pathlib import Path
from .author_loader import ROOT, fingerprint, verify_originals
from .atomic_io import read_json_file


def show_if_completed():
    pointer_path = ROOT/'artifacts/workflow_state/corrected_node_suite_current.json'
    if not pointer_path.exists():
        return None
    pointer = read_json_file(pointer_path)
    ledger = read_json_file(pointer['ledger'])
    if ledger['state'] != 'completed_four_task_repeat_grid':
        return None
    results = read_json_file(Path(pointer['directory'])/'results.json')
    if not all(group['complete_requested_grid'] for group in results['groups']):
        raise ValueError('Completed ledger disagrees with report coverage')
    verify_originals()
    for record in ledger['records'].values():
        if record['state'] != 'completed':
            raise ValueError('An unfinished record cannot be shown as complete')
        for path,digest in record['artifact_sha256'].items():
            if fingerprint(path) != digest:
                raise ValueError('A completed artifact changed: '+path)
    print('FOUR_TASKS_ALREADY_COMPLETE=430/430; historical results retained, no new GPU training',flush=True)
    for group in results['groups']:
        print(f"{group['dataset']}: {group['mean_test_pct']:.4f}% ({group['completed']}/{group['expected']}), paper gap={group['gap_pp']:.4f}pp",flush=True)
    print('RESULTS='+str(Path(pointer['directory'])/'results.json'),flush=True)
    return 0
