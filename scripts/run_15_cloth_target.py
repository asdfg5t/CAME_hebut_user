"""Cloth作业08：离线验收、一次完整确认；达到2个百分点目标后自动完成十seed。"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)


def main():
    from cloth_target_delivery import run, worker, verify, read, OUT, PAUSE
    from came_repro.atomic_io import atomic_write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--stage', choices=['preflight', 'train', 'test'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--resume-head', type=Path)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    if args.pause:
        atomic_write_json(PAUSE, dict(reason='user_requested_cloth_target_pause',
            requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print('已请求安全暂停。下次运行同一文件从已提交完整轮/测试游标续跑。', flush=True)
        return
    verify()
    if args.stage:
        worker(args);return
    print('Cloth作业08：最高已完成验证候选已锁定；官方全验证/全测试/150负例保持。', flush=True)
    print('验收→seed0完整测试一次→距28.88%不超过2个百分点才继续seed1至9。', flush=True)
    print('所有seed保留；未达标会停止并保存，不自动换策略试测试集。', flush=True)
    print('日志和结果：'+str(OUT), flush=True)
    if args.dry_run:
        print('DRY_RUN：配置/源码/选定输入哈希通过；GPU任务0。', flush=True);return
    for name in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE']:
        os.environ[name] = '1'
    if args.detach:
        import psutil
        from came_repro.ide_workflow import acquire_gpu_lock
        OUT.mkdir(parents=True, exist_ok=True)
        with_guard = acquire_gpu_lock(ROOT/'.runtime/cloth_target_launch.lock')
        try:
            launch = OUT/'launch.json'
            if launch.exists():
                saved = read(launch)
                try:
                    p = psutil.Process(saved['pid'])
                    if abs(p.create_time()-saved['creation_time']) < 1:
                        print('本入口已经运行，未启动重复任务。', flush=True);return
                except psutil.NoSuchProcess: pass
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            log = OUT/('console_'+stamp+'.log')
            arguments = [sys.executable, '-u', str(Path(__file__).resolve())]
            if args.check_only: arguments.append('--check-only')
            with log.open('ab') as output:
                child = subprocess.Popen(arguments, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, close_fds=True,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                    env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONUNBUFFERED':'1'})
            atomic_write_json(launch, dict(pid=child.pid, creation_time=psutil.Process(child.pid).create_time(),
                console_log=str(log), check_only=args.check_only, one_shot_not_recurring=True))
            print('已启动离线流程，PID='+str(child.pid)+'；日志='+str(log), flush=True)
        finally: with_guard.close()
        return
    result = run(args.check_only)
    print('Cloth状态：'+result['state'], flush=True)


if __name__ == '__main__':
    main()
