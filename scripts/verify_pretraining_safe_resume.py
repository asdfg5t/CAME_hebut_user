"""Bounded real-data CUDA audit: old step 12 -> safe pause 14 -> resume 16."""
import gc
import json
import os
from pathlib import Path
import sys
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from came_repro import subset_offline_train as subset
from came_repro.author_loader import ROOT,fingerprint
from came_repro.ide_workflow import read_json,write_json
from came_repro.pause_control import TrainingPaused


def main():
    initial=ROOT/'runs/20261011T095723Z_subset_offline_cuda_calibration_seed42_d14bb8'
    checkpoint=Path(read_json(initial/'manifest.json')['checkpoint_directory'])/'last.pt'
    config=ROOT/'artifacts/audit/subset_offline_calibration_config.json'
    request=ROOT/'artifacts/audit/pretraining_pause_verify.request.json'
    original_experiment=subset.SubsetExperiment
    original_arguments=sys.argv
    previous_environment=os.environ.get('CAME_PAUSE_FILE')
    if request.exists():
        request.unlink()
    class AuditPauseExperiment(original_experiment):
        def metric(self,values):
            super().metric(values)
            if values['step']>=14:
                write_json(request,{'purpose':'Two-update real CUDA safe-pause verification'})
    before={path.name for path in (ROOT/'runs').iterdir()}
    try:
        os.environ['CAME_PAUSE_FILE']=str(request)
        subset.SubsetExperiment=AuditPauseExperiment
        sys.argv=['verify','--config',str(config),'--resume',str(checkpoint),'--max-steps','16']
        try:
            subset.main()
            raise AssertionError('Expected coherent pause after step 14')
        except TrainingPaused:
            pass
        new=[path for path in (ROOT/'runs').iterdir() if path.name not in before and path.is_dir()]
        assert len(new)==1
        paused=new[0]
        assert read_json(paused/'status.json')['status']=='paused'
        parent=Path(read_json(paused/'manifest.json')['checkpoint_directory'])/'last.pt'
        state=torch.load(parent,map_location='cpu',weights_only=False)
        assert state['step']==14 and state['schedule']['epoch']==1
        assert state['completed_epochs']==0 and state['completed_subset_rounds']==1
        cursor=state['schedule']['cursors']
        assert 'rng' in state and 'optimizer' in state and 'fixed_seed_pools' in state['schedule']
        del state
        gc.collect()
        request.unlink()
        subset.SubsetExperiment=original_experiment
        before={path.name for path in (ROOT/'runs').iterdir()}
        sys.argv=['verify','--config',str(config),'--resume',str(parent),'--max-steps','16']
        subset.main()
        new=[path for path in (ROOT/'runs').iterdir() if path.name not in before and path.is_dir()]
        assert len(new)==1
        resumed=new[0]
        rows=[json.loads(line) for line in (resumed/'logs/metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows]==[15,16]
        assert read_json(resumed/'manifest.json')['parent_sha256']==fingerprint(parent)
        write_json(ROOT/'artifacts/audit/pretraining_safe_resume_verification.json',{'initial_run':str(initial),'paused_run':str(paused),
                   'resumed_run':str(resumed),'paused_step':14,'resumed_logged_steps':[15,16],'saved_subset_rounds':1,
                   'paused_cursor':cursor,'parent_sha256':fingerprint(parent),'real_cuda':True,'paper_reproduction_complete':False})
        print('SAFE_PRETRAINING_RESUME_VERIFIED',flush=True)
    finally:
        subset.SubsetExperiment=original_experiment
        sys.argv=original_arguments
        if previous_environment is None:
            os.environ.pop('CAME_PAUSE_FILE',None)
        else:
            os.environ['CAME_PAUSE_FILE']=previous_environment
        if request.exists():
            request.unlink()


if __name__=='__main__':
    main()
