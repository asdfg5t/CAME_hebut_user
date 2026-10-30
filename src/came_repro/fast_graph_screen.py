"""Graph-protocol correction first: reuse encoder, freeze on validation, confirm once."""
import argparse
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

from .author_loader import ROOT, fingerprint, verify_originals
from .atomic_io import read_json_file, atomic_write_json
from .ide_workflow import acquire_gpu_lock
from .quality_pipeline import reference_assets, run_child, best_resume

PAPER_TEST_PCT = {'wiki-cs':78.57,'ogbn-arxiv':71.66,'books-nc':83.21}


def choose_policy(reference_validation, candidate_validation, threshold_pp):
    return 'bidirected' if 100*(candidate_validation-reference_validation)>=threshold_pp else 'original'


def validate_profile(profile):
    if profile['datasets']!=['wiki-cs','ogbn-arxiv','books-nc'] or profile['seed']!=0 or profile['wiki_split_id']!=0:
        raise ValueError('Keep the predeclared single-seed scope; Ele-fashion excluded')
    if profile['probe_epochs']!=2000 or profile['probe_patience']!=100 or profile['extra_pretraining_updates']!=0:
        raise ValueError('This shortcut fixes topology without shortening head convergence or retraining the encoder')
    if profile['cumulative_time_limit'] is not None or profile['min_validation_gain_pp']<0:
        raise ValueError('No cumulative deadline; explicit validation threshold required')
    if profile['owned_cache_limit_gib']<=0 or any(v<1 for v in profile['inference_batches'].values()):
        raise ValueError('Invalid storage/inference budget')


def artifacts_for(run):
    run=Path(run)
    names=['summary.json','manifest.json','config.json','status.json','logs/metrics.csv','logs/metrics.jsonl','figures/probe_curves.png']
    names += ['test_predictions.npz'] if read_json_file(run/'summary.json')['test_evaluations'] else ['selection_predictions.npz']
    return {str(run/name):fingerprint(run/name) for name in names}


def wiki_confirmation(model_hash):
    lock_path=ROOT/'artifacts/audit/wiki_fast_confirmation_lock.json'
    lock=read_json_file(lock_path)
    if lock['encoder_sha256']!=model_hash or lock['test_used_for_selection'] or choose_policy(lock['reference_validation_accuracy'],lock['candidate_validation_accuracy'],.5)!='bidirected':
        raise ValueError('Invalid Wiki-CS selection lock')
    matches=[]
    for run in (ROOT/'runs').glob('*_quality_came_*'):
        if not (run/'summary.json').is_file():
            continue
        result=read_json_file(run/'summary.json')
        if (result.get('dataset')=='wiki-cs' and result.get('graph_policy')=='bidirected' and result.get('seed')==0
                and result.get('split_id')==0 and result.get('encoder_sha256')==model_hash and result.get('test_evaluations')==1):
            manifest=read_json_file(run/'manifest.json')
            stamp=datetime.datetime.strptime(manifest['created_utc'],'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc)
            if stamp>=datetime.datetime.fromisoformat(lock['locked_utc']):
                matches.append((manifest['created_utc'],run,result))
    if not matches:
        raise ValueError('The frozen Wiki-CS confirmation must finish first')
    # First matching successful run, never the highest test score.
    _,run,result=min(matches,key=lambda value:value[0])
    return {'state':'completed','run_directory':str(run),'summary':result,'artifact_sha256':artifacts_for(run),'reused':True,'selection_lock_sha256':fingerprint(lock_path)}


def report(ledger,directory):
    rows=[]
    for dataset in ledger['identity']['profile']['datasets']:
        reference=ledger['references'][dataset]
        candidate=ledger['records'].get('validate_'+dataset,{})
        confirmation=ledger['records'].get('confirm_'+dataset,{})
        test=confirmation.get('summary',{}).get('test_accuracy')
        rows.append({'dataset':dataset,'reference_validation_pct':100*reference['validation_accuracy'],
                     'candidate_validation_pct':100*candidate['summary']['validation_accuracy'] if candidate.get('state')=='completed' else None,
                     'selected_policy':ledger.get('selections',{}).get(dataset,{}).get('policy'),
                     'test_pct':100*test if test is not None else None,'paper_test_pct':PAPER_TEST_PCT[dataset],
                     'gap_pp':100*test-PAPER_TEST_PCT[dataset] if test is not None else None,
                     'test_run':confirmation.get('run_directory')})
    atomic_write_json(directory/'results.json',{'rows':rows,'state':ledger['state'],'scope':'one seed; Wiki-CS split 0; paper values are multi-run statistics',
                      'test_used_for_selection':False,'extra_encoder_updates':0,'paper_reproduction_complete':False})
    import csv
    with (directory/'results.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(len(rows));fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for offset,key,label in [(-.18,'reference_validation_pct','Original validation'),(.18,'candidate_validation_pct','Bidirected validation')]:
        axes[0].bar(x+offset,[r[key] if r[key] is not None else np.nan for r in rows],.36,label=label)
    for offset,key,label in [(-.18,'test_pct','One-seed confirmation'),(.18,'paper_test_pct','Paper mean (reference)')]:
        axes[1].bar(x+offset,[r[key] if r[key] is not None else np.nan for r in rows],.36,label=label)
    for axis,title in zip(axes,['Validation selects policy','Held-out test confirms once']):
        axis.set(xticks=x,xticklabels=[r['dataset'] for r in rows],ylim=(0,100),ylabel='Accuracy (%)',title=title)
        axis.tick_params(axis='x',labelsize=8);axis.legend(fontsize=7)
    fig.text(.02,.01,'Source: results.json and saved predictions; paper Table 1. One seed/split is not a full reproduction; blank bars = pending.',fontsize=7)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(directory/'comparison.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,default=ROOT/'configs/fast_graph_screen.json')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();profile=read_json_file(args.profile);validate_profile(profile)
    model,_,_,_=reference_assets();model_hash=fingerprint(model)
    old_pointer=read_json_file(ROOT/'artifacts/workflow_state/quality_pilot_current.json')
    old=read_json_file(old_pointer['ledger'])
    references={d:old['records']['reference_'+d]['summary'] for d in profile['datasets']}
    reused=wiki_confirmation(model_hash)
    if args.dry_run:
        print(json.dumps({'encoder_retraining':False,'ele_fashion_retraining':False,'wiki_test_pct':100*reused['summary']['test_accuracy'],
                          'remaining_domains':profile['datasets'][1:],'head_epochs_max':2000,'cumulative_deadline':None,'training_started':False},indent=2));return 0
    if Path(sys.executable).resolve()!=(ROOT/'.venv-came39/Scripts/python.exe').resolve():
        raise RuntimeError('Verified project Python required')
    lock=acquire_gpu_lock(ROOT/'.runtime/fast_graph_screen.lock');logger=None
    try:
        gpu=acquire_gpu_lock();gpu.close();verify_originals()
        sources={str(p.relative_to(ROOT)):fingerprint(p) for pattern in ('src/came_repro/*.py','scripts/*.py') for p in ROOT.glob(pattern)}
        data={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'data/processed').glob('*/*/manifest.json')}
        import torch,dgl
        identity={'profile':profile,'sources':sources,'data':data,'encoder_sha256':model_hash,
                  'environment':{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'dgl':dgl.__version__,'cuda':torch.version.cuda},
                  'originals':verify_originals()}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
        state=ROOT/'artifacts/workflow_state';directory=ROOT/'artifacts/fast_graph_screen'/key;directory.mkdir(parents=True,exist_ok=True)
        pointer_path=state/'fast_graph_screen_current.json'
        if pointer_path.exists() and read_json_file(pointer_path)['identity_key']!=key:
            raise ValueError('Source/data/profile changed; retain and review the previous fast ledger')
        path=state/('fast_graph_screen_'+key+'.json');pause=state/('fast_graph_screen_'+key+'_pause.json')
        ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'ready','active_seconds':0.,'references':references,'records':{},'selections':{}}
        for record in ledger['records'].values():
            if record['state']=='completed':
                if any(fingerprint(p)!=h for p,h in record['artifact_sha256'].items()):
                    raise ValueError('Completed artifact changed')
        if pause.exists():pause.unlink()
        atomic_write_json(pointer_path,{'identity_key':key,'ledger':str(path),'pause_file':str(pause),'directory':str(directory)})
        atomic_write_json(directory/'provenance.json',identity)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        logger=logging.getLogger(str(directory)+stamp);logger.setLevel(logging.INFO)
        for handler in (logging.StreamHandler(),logging.FileHandler(directory/('pipeline_'+stamp+'.log'),encoding='utf-8')):
            formatter=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');formatter.converter=time.gmtime
            handler.setFormatter(formatter);logger.addHandler(handler)
        ledger['records'].setdefault('confirm_wiki-cs',reused)
        ledger['records'].setdefault('validate_wiki-cs',old['records']['graph_wiki-cs'])
        ledger['selections'].setdefault('wiki-cs',{'policy':'bidirected','locked_before_test':True,'selection_source':'wiki_fast_confirmation_lock.json'})
        ledger['state']='running'
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        def execute(dataset,confirmation,policy):
            jid=('confirm_' if confirmation else 'validate_')+dataset
            record=ledger['records'].setdefault(jid,{'state':'pending','attempts':[]})
            if record['state']=='completed':return True
            if pause.exists():ledger['state']='paused';return False
            if any(fingerprint(ROOT/p)!=h for p,h in sources.items()):raise ValueError('Code changed during a trial')
            command=['-m','came_repro.quality_probe','--dataset',dataset,'--representation','came','--checkpoint',str(model),
                     '--graph-policy',policy,'--seed','0','--split-id','0','--epochs','2000','--patience','100',
                     '--inference-batch-size',str(profile['inference_batches'][dataset]),'--validation-cache-mib',str(profile['validation_cache_mib']),
                     '--owned-cache-limit-gib',str(profile['owned_cache_limit_gib'])]
            resume=best_resume(record,'probe_last.pt')
            if resume:command+=['--resume-head',str(resume)]
            if confirmation:command+=['--confirm-test','--discard-derived-cache']
            logger.info('FAST_JOB=%s policy=%s; existing encoder retained',jid,policy)
            code,run=run_child(command,pause,ledger,path,record,logger,float('inf'))
            if code!=0 or not run:
                record['state']='paused' if pause.exists() or code==75 else 'failed';ledger['state']=record['state'];return False
            result=read_json_file(Path(run)/'summary.json')
            if result['test_evaluations']!=int(confirmation):raise ValueError('Unexpected test access')
            record.update(state='completed',summary=result,run_directory=run,artifact_sha256=artifacts_for(run))
            atomic_write_json(path,ledger);report(ledger,directory);return True
        for dataset in profile['datasets'][1:]:
            if not execute(dataset,False,'bidirected'):break
            result=ledger['records']['validate_'+dataset]['summary']
            policy=choose_policy(references[dataset]['validation_accuracy'],result['validation_accuracy'],profile['min_validation_gain_pp'])
            frozen={'policy':policy,'reference_validation_accuracy':references[dataset]['validation_accuracy'],
                    'candidate_validation_accuracy':result['validation_accuracy'],'locked_before_test':True,'test_used_for_selection':False}
            existing=ledger['selections'].get(dataset)
            if existing is not None and existing!=frozen:raise ValueError('Cannot change a frozen policy')
            ledger['selections'][dataset]=frozen
            atomic_write_json(directory/(dataset+'_selection_lock.json'),frozen);atomic_write_json(path,ledger)
            if policy=='original':
                # No improvement: keep the original completed result; do not
                # repeatedly test or select by the paper target.
                full=read_json_file(read_json_file(ROOT/'artifacts/workflow_state/full_suite_current.json')['ledger'])
                prior=full['records']['eval_came_42_node_'+dataset+'_0_0']
                ledger['records']['confirm_'+dataset]={'state':'completed','summary':prior['summary'],'run_directory':prior['run_directory'],
                                                     'artifact_sha256':prior['artifact_sha256'],'reused':True}
                from .quality_probe import owned_cache_path,remove_owned_cache
                storage=read_json_file(ROOT/'configs/storage.json')
                remove_owned_cache(owned_cache_path(storage,model_hash,dataset,'bidirected'),storage)
            elif not execute(dataset,True,policy):break
        else:
            ledger['state']='completed_single_seed_graph_screen'
        atomic_write_json(path,ledger);report(ledger,directory)
        print('FAST_STATE='+ledger['state'],flush=True);print('FAST_RESULTS='+str(directory/'results.json'),flush=True)
        return 1 if ledger['state']=='failed' else 0
    except BaseException as error:
        if 'ledger' in locals():
            ledger['state']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed';ledger['error']=str(error);atomic_write_json(path,ledger)
        raise
    finally:
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if logger:
            for handler in list(logger.handlers):handler.close();logger.removeHandler(handler)
        lock.close()


if __name__=='__main__':
    raise SystemExit(main())
