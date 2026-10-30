"""Bounded, predeclared WN/Sports/Cloth validation repairs; no test-based selection."""
import copy
import datetime
import logging
from pathlib import Path
from .atomic_io import read_json_file,atomic_write_json
from .author_loader import ROOT,fingerprint,verify_originals
from .ide_workflow import acquire_gpu_lock
from .quality_pipeline import run_child,best_resume
from .remaining_kg_validation import artifact_hashes,select_candidate,verify_records
from .storage_budget import check_budget


def choose_lp_candidate(candidates,reference,min_gain_pp=.5):
    best=max(candidates,key=lambda r:r['summary']['validation_mrr'])
    return best if 100*(best['summary']['validation_mrr']-reference['summary']['validation_mrr'])>=min_gain_pp else reference


def main():
    verify_originals();profile_path=ROOT/'configs/validation_repair_20261014.json';profile=read_json_file(profile_path)
    if profile['seed']!=0 or profile['test_used_for_selection'] is not False:raise ValueError('Declared validation-only plan required')
    kg=read_json_file(ROOT/'artifacts/workflow_state/kg_affine_screen_current.json')
    lp=read_json_file(ROOT/'artifacts/workflow_state/remaining_lp_validation_current.json')
    if kg['state']!='completed_two_kg_diagnostics' or lp['state']!='paused':raise RuntimeError('Finish/pause earlier owned diagnostics before repair dispatch')
    verify_records(kg);verify_records(lp)
    model=Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(model)!=profile['encoder_sha256']:raise ValueError('Frozen encoder changed')
    storage=read_json_file(ROOT/'configs/storage.json')
    cache_root=Path(storage['embedding_cache_root'])
    sources={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    identity={'profile':profile,'sources':sources,'encoder_sha256':fingerprint(model),
        'references':{k:fingerprint(ROOT/p) for k,p in {'KG':'artifacts/workflow_state/kg_affine_screen_current.json','LP':'artifacts/workflow_state/remaining_lp_validation_current.json'}.items()}}
    path=ROOT/'artifacts/workflow_state/validation_repair_current.json'
    ledger=read_json_file(path) if path.is_file() else {'identity':identity,'state':'ready','records':{},'active_seconds':0.,'selections':{}}
    if ledger['identity']!=identity:raise ValueError('Repair identity changed; audit before resume')
    verify_records(ledger)
    directory=ROOT/'artifacts/remaining_main_pilots/validation_repair';directory.mkdir(parents=True,exist_ok=True)
    lock=acquire_gpu_lock(ROOT/'.runtime/validation_repair.lock')
    logger=logging.getLogger('validation_repair');logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(),logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
    pause=ROOT/'artifacts/workflow_state/validation_repair_pause.json'
    def execute(jid,args,resume_name='probe_last.pt'):
        for name,digest in sources.items():
            if fingerprint(ROOT/name)!=digest:raise ValueError('Active repair source changed')
        record=ledger['records'].setdefault(jid,{'state':'ready','attempts':[]})
        if record['state']=='completed':return record
        resume=best_resume(record,resume_name)
        if resume:args+=['--resume-head',str(resume)]
        ledger['state']='running'
        code,run=run_child(args,pause,ledger,path,record,logger,float('inf'))
        if code or not run:
            record['state']='paused' if code==75 else 'failed';ledger['state']=record['state'];atomic_write_json(path,ledger)
            raise RuntimeError('Repair '+jid+' exited '+str(code))
        artifacts=artifact_hashes(run)
        for name in ('test_link_ranks.npz','validation_link_ranks.npz','reverse_overlap_queries.npz'):
            file=Path(run)/name
            if file.is_file():artifacts[str(file)]=fingerprint(file)
        record.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=artifacts)
        atomic_write_json(path,ledger);return record
    def freeze_and_confirm(dataset,selection,module):
        if dataset in ledger['selections'] and ledger['selections'][dataset]!=selection:
            raise ValueError('Cannot replace a frozen repair selection')
        selected_path=directory/(dataset+'_selection_lock.json')
        ledger['selections'][dataset]=selection
        atomic_write_json(selected_path,selection);atomic_write_json(path,ledger)
        result=execute('confirm_'+dataset,['-m',module,'--selection-lock',str(selected_path)],'__no_training_resume__')
        logger.info('CONFIRMED dataset=%s summary=%s',dataset,str(result['summary']))
    try:
        ledger['budget_at_start']=check_budget();atomic_write_json(path,ledger)
        wn_cache=cache_root/'quality_pilot'/fingerprint(model)[:20]/'WN18RR_train_reverse'
        candidates=[execute('WN_LBFGS_l2_'+str(l2),['-m','came_repro.kg_linear_repair','--checkpoint',str(model),'--embedding-cache',str(wn_cache),'--l2',str(l2)]) for l2 in profile['WN']['l2_candidates']]
        reference=kg['records']['confirm_WN18RR'];selected=select_candidate(candidates,reference)
        if selected is reference:
            ledger['records']['confirm_WN18RR']=copy.deepcopy(reference);ledger['selections']['WN18RR']={'reuse_reference':True,'validation_accuracy':reference['summary']['validation_accuracy']}
            atomic_write_json(path,ledger)
        else:
            run=Path(selected['run_directory']);s=selected['summary'];best=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])/'probe_best_raw_linear.pt'
            selection={'dataset':'WN18RR','encoder_sha256':fingerprint(model),'graph_policy':'train_reverse','feature_transform':'train_pair_affine',
                'classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),
                'validation_summary_sha256':fingerprint(run/'summary.json'),'selected_validation_accuracy':s['validation_accuracy'],
                'embedding_cache':str(wn_cache),'locked_before_test':True,'test_used_for_selection':False,
                'reference_validation_accuracy':reference['summary']['validation_accuracy'],'candidate_validation_accuracies':[r['summary']['validation_accuracy'] for r in candidates]}
            freeze_and_confirm('WN18RR',selection,'came_repro.kg_selected_confirmation')
        for dataset in ('sports-copurchase','cloth-copurchase'):
            reference=lp['records'][dataset]
            if reference['state']!='completed':raise ValueError('Complete baseline required for '+dataset)
            cache=cache_root/fingerprint(model)[:20]/dataset
            candidates=[execute(dataset+'_'+variant,['-m','came_repro.lp_validation_repair','--dataset',dataset,'--checkpoint',str(model),
                '--embedding-cache',str(cache),'--variant',variant],'link_head_last.pt') for variant in profile['LP']['variants']]
            selected=choose_lp_candidate(candidates,reference)
            if selected is reference:
                ledger['records']['confirm_'+dataset]=copy.deepcopy(reference);ledger['selections'][dataset]={'reuse_reference':True,'validation_mrr':reference['summary']['validation_mrr']}
                atomic_write_json(path,ledger)
            else:
                run=Path(selected['run_directory']);s=selected['summary'];best=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
                selection={'dataset':dataset,'encoder_sha256':fingerprint(model),'head_variant':s['head_variant'],
                    'classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),
                    'validation_summary_sha256':fingerprint(run/'summary.json'),'selected_validation_mrr':s['validation_mrr'],
                    'embedding_cache':str(cache),'locked_before_test':True,'test_used_for_selection':False,
                    'reference_validation_mrr':reference['summary']['validation_mrr'],'candidate_validation_mrr':[r['summary']['validation_mrr'] for r in candidates]}
                freeze_and_confirm(dataset,selection,'came_repro.lp_selected_confirmation')
        ledger['state']='completed_bounded_validation_repairs';ledger['baseline_books_lp_still_paused']=True
        atomic_write_json(path,ledger)
    except BaseException as error:
        if ledger['state']!='paused':ledger['state']='failed'
        ledger['error']=str(error);atomic_write_json(path,ledger);raise
    finally:lock.close()

if __name__=='__main__':main()

