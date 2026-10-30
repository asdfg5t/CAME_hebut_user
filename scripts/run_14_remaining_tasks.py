"""剩余主任务06/08/09/10交付入口；PyCharm直接运行，重复点击从断点继续。"""
import argparse
import datetime
import sys
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    from remaining_delivery import verify_release,run,read,PROFILE,PAUSE,OUT
    from came_repro.atomic_io import atomic_write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--pause',action='store_true')
    args=parser.parse_args()
    if args.pause:
        atomic_write_json(PAUSE,dict(reason='user_requested_remaining_delivery_pause',requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print('已请求安全暂停。完整轮/编码游标提交后退出；下次点同一文件会续跑。',flush=True);return
    verify_release()
    print('剩余主任务06 WN、08 Cloth、09 Goodreads-LP、10 Products；目标40个下游seed结果。',flush=True)
    print('保留原450结果。验证集选模型、所有固定seed均保留；未保证距论文2pp。',flush=True)
    print('日志/结果：'+str(OUT),flush=True)
    if args.dry_run:
        p=read(PROFILE);print('DRY_RUN：配置与交付源码哈希通过；GPU作业0；不下载、不重训、不删除。',flush=True)
        print('种子：'+str(p['seeds'])+'；现有诊断活动时会等待，Products必须先过精确数值和容量门槛。',flush=True);return
    try:
        result=run(args.preflight_only);print('当前交付状态：'+result['state'],flush=True)
    except KeyboardInterrupt:print('已中断入口；原子检查点保留。下次运行从已提交游标/完整轮继续。',flush=True)
    except Exception as error:
        print('当前阶段没有完成：'+str(error),flush=True)
        print('已完成结果保留。具体状态与错误已写入交付账本及日志，不会回退CPU或缩减官方评测。',flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
