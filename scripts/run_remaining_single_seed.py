"""After NC review: measure the six remaining paper tasks once, with safe resume."""
import sys
from pathlib import Path
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import hashlib
import logging
import datetime
import time
from came_repro.author_loader import ROOT,fingerprint,verify_originals
from came_repro.atomic_io import read_json_file,atomic_write_json
from came_repro.ide_workflow import acquire_gpu_lock
from came_repro.quality_pipeline import reference_assets,run_child,best_resume,remaining_first_seed_jobs,evict_first_seed_cache


def main():
    fast=read_json_file(read_json_file(ROOT/'artifacts/workflow_state/fast_graph_screen_current.json')['ledger'])
    if fast['state']!='completed_single_seed_graph_screen':
        raise RuntimeError('Finish and review run_fast_graph_screen.py first')
    model,_,_,_=reference_assets()
    jobs=remaining_first_seed_jobs('reference')
    sources={str(p.relative_to(ROOT)):fingerprint(p) for pattern in ('src/came_repro/*.py','scripts/*.py') for p in ROOT.glob(pattern)}
    identity={'encoder_sha256':fingerprint(model),'jobs':jobs,'sources':sources,'originals':verify_originals(),
              'scope':'six remaining tasks, seed zero; no automatic full repeats or ablations'}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    state=ROOT/'artifacts/workflow_state';path=state/('remaining_single_seed_'+key+'.json');pause=state/('remaining_single_seed_'+key+'_pause.json')
    pointer=state/'remaining_single_seed_current.json'
    if pointer.exists() and read_json_file(pointer)['identity_key']!=key:raise ValueError('Changed source/model; review the old ledger')
    directory=ROOT/'artifacts/remaining_single_seed'/key;directory.mkdir(parents=True,exist_ok=True)
    ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'ready','active_seconds':0.,'records':{}}
    for record in ledger['records'].values():
        if record['state']=='completed' and any(fingerprint(p)!=h for p,h in record['artifact_sha256'].items()):raise ValueError('Completed artifact changed')
    lock=acquire_gpu_lock(ROOT/'.runtime/remaining_single_seed.lock')
    logger=logging.getLogger(str(directory));logger.setLevel(logging.INFO)
    handler=logging.FileHandler(directory/'pipeline.log',encoding='utf-8');formatter=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');formatter.converter=time.gmtime
    handler.setFormatter(formatter);logger.addHandler(handler);logger.addHandler(logging.StreamHandler())
    try:
        gpu=acquire_gpu_lock();gpu.close()
        if pause.exists():pause.unlink()
        atomic_write_json(pointer,{'identity_key':key,'ledger':str(path),'pause_file':str(pause),'directory':str(directory)})
        ledger['state']='running'
        for job in jobs:
            record=ledger['records'].setdefault(job['id'],{'job':job,'state':'pending','attempts':[]})
            if record['state']=='completed':continue
            if pause.exists():ledger['state']='paused';break
            if any(fingerprint(ROOT/p)!=h for p,h in sources.items()):raise ValueError('Code changed during execution')
            module='came_repro.link_evaluation' if job['task']=='link' else 'came_repro.downstream'
            command=['-m',module,'--dataset',job['dataset'],'--representation','came','--seed','0','--checkpoint',str(model),
                     '--inference-batch-size','4' if job['dataset']=='ogbn-products' else '32']
            if job['task']=='link':
                command+=['--epochs','100','--patience','10'];resume=best_resume(record,'link_head_last.pt')
            else:
                command+=['--task',job['task'],'--epochs','2000','--patience','100'];resume=best_resume(record,'probe_last.pt')
            if resume:command+=['--resume-head',str(resume)]
            code,run=run_child(command,pause,ledger,path,record,logger,float('inf'))
            if code!=0 or not run:record['state']='paused' if pause.exists() or code==75 else 'failed';ledger['state']=record['state'];break
            run=Path(run);summary=read_json_file(run/'summary.json')
            if not read_json_file(run/'status.json')['status'].startswith('completed_') or summary['test_evaluations']!=1:raise ValueError('Task did not complete correctly')
            files=list(run.glob('*.npz'))+[run/n for n in ('summary.json','config.json','manifest.json','status.json','logs/metrics.csv','logs/metrics.jsonl')]
            files+=list((run/'figures').glob('*.png'))
            record.update(state='completed',run_directory=str(run),summary=summary,artifact_sha256={str(p):fingerprint(p) for p in files})
            atomic_write_json(path,ledger);evict_first_seed_cache(job,model,directory)
        else:ledger['state']='completed_single_seed_measurements'
        atomic_write_json(path,ledger);atomic_write_json(directory/'results.json',{'state':ledger['state'],'records':ledger['records'],'paper_reproduction_complete':False})
        print('REMAINING_STATE='+ledger['state'],flush=True)
        return 1 if ledger['state']=='failed' else 0
    finally:
        lock.close()
        for h in list(logger.handlers):h.close();logger.removeHandler(h)


if __name__=='__main__':raise SystemExit(main())
