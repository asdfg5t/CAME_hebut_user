"""Launch one Windows process detached from the chat/tool job, without a window."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cloth_wn_autonomous_20261017'


def main():
    if os.name!='nt':raise RuntimeError('This detached launcher is for this Windows workstation')
    sys.path.insert(0,str(ROOT/'src'))
    from came_repro.ide_workflow import acquire_gpu_lock
    guard=acquire_gpu_lock(ROOT/'.runtime/cloth_wn_offline_launch.lock')
    try:
        import psutil
        old=OUT/'launch.json'
        if old.exists():
            saved=json.loads(old.read_text(encoding='utf-8'))
            try:
                process=psutil.Process(saved['pid'])
                if abs(process.create_time()-saved['process_creation_time'])<1:
                    print(json.dumps(dict(already_running=True,**saved),ensure_ascii=True));return
            except psutil.NoSuchProcess:pass
        if (ROOT/'artifacts/workflow_state/representation_coverage_pause.json').exists():
            raise RuntimeError('User pause exists: explicit resume required; nothing launched')
        for lock in ('pycharm_gpu','representation_coverage','cloth_wn_user_entry'):
            probe=acquire_gpu_lock(ROOT/'.runtime'/(lock+'.lock'));probe.close()
        OUT.mkdir(parents=True,exist_ok=True)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        log=OUT/('offline_console_'+stamp+'.log')
        interpreter=ROOT/'.venv-came39/Scripts/python.exe'
        flags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
        with log.open('ab') as output:
            child=subprocess.Popen([str(interpreter),'-u',str(ROOT/'scripts/run_08_cloth_wn_repair.py')],
                cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                creationflags=flags,close_fds=True,
                env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8',
                     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
        creation=psutil.Process(child.pid).create_time()
        saved={'pid':child.pid,'process_creation_time':creation,'console_log':str(log),
               'entry':str(ROOT/'scripts/run_08_cloth_wn_repair.py'),
               'launched_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'detached_from_console':True,'breakaway_from_parent_job_requested_and_accepted':True,
               'offline_environment':True,'not_a_recurring_automation':True}
        from came_repro.atomic_io import atomic_write_json
        atomic_write_json(old,saved)
        print(json.dumps(saved,ensure_ascii=True))
    finally:guard.close()


if __name__=='__main__':main()
