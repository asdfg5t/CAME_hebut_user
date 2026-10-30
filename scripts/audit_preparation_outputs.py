"""Read-only verification of actual smoke-run records against saved source numbers."""
import argparse
import csv
import datetime
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from PIL import Image
from came_repro.author_loader import fingerprint,verify_originals
from came_repro.experiment import write_json
from came_repro.prepared_data import prepared_directory
from came_repro.storage_budget import check_budget


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inspect_run(directory):
    directory=Path(directory).resolve()
    config,summary,manifest,status=[read_json(directory/name) for name in ('config.json','summary.json','manifest.json','status.json')]
    if status['status'] not in ('completed_partial_budget','completed_downstream_task'):
        raise ValueError('Expected a successful bounded preparation run')
    assert summary['paper_reproduction_complete'] is False
    rows=[json.loads(line) for line in (directory/'logs/metrics.jsonl').read_text().splitlines()]
    with (directory/'logs/metrics.csv').open(newline='') as source:
        csv_rows=list(csv.DictReader(source))
    assert len(rows)==len(csv_rows)>0
    for a,b in zip(rows,csv_rows):
        assert set(a)==set(b)
        for key,value in a.items():
            if isinstance(value,(int,float)):
                assert math.isfinite(value) and math.isclose(value,float(b[key]),rel_tol=1e-12,abs_tol=1e-12)
            else:
                assert str(value)==b[key]
    assert (directory/'logs/train.log').stat().st_size>0
    assert list((directory/'tensorboard').glob('events.out.tfevents.*'))
    assert (directory/'source/requirements.freeze.txt').is_file()
    assert manifest['original_files']==verify_originals()
    if manifest['parent_checkpoint']:
        assert fingerprint(manifest['parent_checkpoint'])==manifest['parent_sha256']
    figures={}
    for path in (directory/'figures').glob('*.png'):
        with Image.open(path) as picture:
            picture.verify()
        figures[path.name]=fingerprint(path)
    assert figures
    result={'directory':str(directory),'status':status,'metrics_rows':len(rows),
            'figures_sha256':figures,'summary_sha256':fingerprint(directory/'summary.json')}
    if 'step' in summary:
        state=torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)
        assert state['step']==summary['step'] and state['schedule']['epoch']==summary['completed_epochs']==0
        assert all(key in state for key in ('model','optimizer','rng','schedule','counts'))
        assert state['schedule']['cursors']==summary['seed_coverage_current_epoch']
        assert list(range(rows[0]['step'],rows[-1]['step']+1))==[row['step'] for row in rows]
        result['checkpoint_step']=state['step']
        result['checkpoint_sha256']=fingerprint(summary['checkpoint'])
        del state
    elif summary['task']=='link':
        with np.load(directory/'test_link_ranks.npz') as values:
            assert math.isclose(float(np.mean(1/values['rank'])),summary['test']['official']['mrr'],rel_tol=1e-12)
            assert len(values['rank'])==summary['test']['queries']
            assert int(values['reverse_overlap'].sum())==summary['test']['reverse_overlap_queries']
            for cutoff in (1,3,10):
                assert math.isclose(float(np.mean(values['rank']<=cutoff)),summary['test']['official'][f'hits@{cutoff}'],rel_tol=1e-12)
            clean=~values['reverse_overlap']
            assert math.isclose(float(np.mean(1/values['rank'][clean])),summary['test']['reverse_nonoverlap_subset']['mrr'],rel_tol=1e-12)
        assert fingerprint(directory/'reverse_overlap_queries.npz')==summary['reverse_overlap_indices_sha256']
        assert fingerprint(directory/'test_link_ranks.npz')==summary['test_query_ranks_sha256']
        result['test_ranks_recomputed']=True
    elif summary['task']=='few_shot':
        labels=np.load(prepared_directory(summary['dataset'])/'labels.npy',mmap_mode='r')
        for record in summary['episodes']:
            with np.load(directory/f'episode_{record["shot"]}shot.npz') as values:
                assert not np.intersect1d(values['support_ids'],values['query_ids']).size
                assert values['support_ids'].shape==(5,record['shot'])
                assert float(np.mean(values['prediction']==labels[values['query_ids']]))==record['accuracy']
        result['episodes_recomputed']=True
    else:
        with np.load(directory/'test_predictions.npz') as values:
            assert float(np.mean(values['prediction']==values['labels']))==summary['test_accuracy']
        result['test_predictions_recomputed']=True
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    args=parser.parse_args()
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_files':verify_originals(),
            'runs':[inspect_run(path) for path in args.runs],'storage':check_budget(),
            'full_training_started':False,'paper_reproduction_complete':False,
            'scope':'Actual bounded CUDA entrypoint/restore/log/metric/checkpoint/figure acceptance; full budgets left to user.'}
    write_json(ROOT/'artifacts/audit/preparation_acceptance.json',report)
    print('PREPARATION_OUTPUTS_VERIFIED: '+str(len(report['runs']))+' bounded runs')


if __name__=='__main__':
    main()
