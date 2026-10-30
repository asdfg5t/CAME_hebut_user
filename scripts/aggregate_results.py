"""Aggregate explicitly selected, comparable runs; never select runs by test performance."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from came_repro.experiment import write_json


def metrics(summary):
    if summary['task']=='few_shot':
        return {str(row['shot'])+'shot_accuracy':row['accuracy'] for row in summary['episodes']}
    if summary['task']=='link':
        result={'official_'+key:value for key,value in summary['test']['official'].items()}
        clean=summary['test']['reverse_nonoverlap_subset']
        if clean:
            result.update({'reverse_nonoverlap_'+key:value for key,value in clean.items()})
        return result
    return {'test_accuracy':summary['test_accuracy']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-seeds',type=int,default=10)
    parser.add_argument('--expected-splits',type=int,default=1)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    if args.expected_seeds < 1 or args.expected_splits < 1:
        raise ValueError('Expected seeds/splits must be positive')
    groups,identity,records={},None,[]
    for directory in args.runs:
        summary=json.loads((directory/'summary.json').read_text())
        config=json.loads((directory/'config.json').read_text())
        manifest=json.loads((directory/'manifest.json').read_text())
        if json.loads((directory/'status.json').read_text())['status']!='completed_downstream_task':
            raise ValueError('Only completed downstream task runs may be aggregated')
        code={name:sha for name,sha in manifest['harness_sources'].items() if name.replace('\\','/').startswith('src/came_repro/')}
        key=(summary['dataset'],summary['task'],summary['representation'],config.get('variant','came'),
             json.dumps(config.get('model'),sort_keys=True),json.dumps(config.get('sampling'),sort_keys=True),
             json.dumps(config.get('training'),sort_keys=True),json.dumps(config.get('deviations'),sort_keys=True),
             config.get('evaluation',{}).get('epochs'),config.get('evaluation',{}).get('patience'),
             config.get('evaluation',{}).get('selection_split'),config.get('evaluation',{}).get('probe_batch_size'),
             json.dumps(code,sort_keys=True),manifest['data'].get('prepared_manifest_sha256'))
        if identity is not None and key!=identity:
            raise ValueError('Runs differ in dataset/task/model/protocol; aggregate each profile separately')
        identity=key
        if not args.allow_partial and summary['representation']=='came' and not summary.get('pretraining_complete'):
            raise ValueError('Incomplete pretraining cannot enter an intended full-budget comparison')
        required_epochs=2000 if summary['task'] in ('node','relation') else 100 if summary['task']=='link' else None
        if not args.allow_partial and required_epochs is not None and summary.get('max_epochs')!=required_epochs:
            raise ValueError('Reduced head budget requires --allow-partial')
        split_id=config.get('evaluation',{}).get('split_id',0)
        seed=config['seed']
        group=groups.setdefault(split_id,{})
        if seed in group:
            raise ValueError('Duplicate seed in the same split; do not select the better test result')
        group[seed]=metrics(summary)
        records.append({'run':str(directory.resolve()),'seed':seed,'split_id':split_id,
                        'encoder_sha256':config.get('evaluation',{}).get('encoder_sha256'),
                        'code_sha256':manifest['harness_sources'],**group[seed]})
    complete=len(groups)==args.expected_splits and all(len(value)==args.expected_seeds for value in groups.values())
    seed_sets=[set(value) for value in groups.values()]
    complete=complete and all(value==seed_sets[0] for value in seed_sets)
    if not complete and not args.allow_partial:
        raise ValueError('Expected seed/split grid is incomplete')
    stats={}
    for split_id,values in groups.items():
        stats[split_id]={name:{'mean':float(np.mean([row[name] for row in values.values()])),
                               'sample_std':float(np.std([row[name] for row in values.values()],ddof=1)) if len(values)>1 else None,
                               'count':len(values)} for name in next(iter(values.values()))}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'aggregate.json',{'identity':identity,'per_split':stats,'runs':records,'complete_requested_grid':complete,
                                            'allow_partial':args.allow_partial,'paper_reproduction_complete':False,
                                            'scale':'Accuracy/MRR/Hits are fractions; std uses ddof=1; splits and seeds remain separate.'})
    first_split=next(iter(groups.values()))
    columns=['run','seed','split_id','encoder_sha256']+list(next(iter(first_split.values())))
    with (args.output/'seed_metrics.csv').open('w',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print('AGGREGATE_DIR='+str(args.output.resolve()))


if __name__=='__main__':
    main()
