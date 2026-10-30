"""主任务07：Sports固定协议十下游seed；保留已验证seed0，仅补1-9。"""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
    root = Path(__file__).resolve().parents[1]
    print('[CAME] Main task07 / Sports: retain seed0; train downstream seeds1-9.', flush=True)
    print('[CAME] Verified seed0 MRR=35.695525%; paper=37.68%; gap=-1.984475pp.', flush=True)
    print('[CAME] Full repeats are still unmeasured; seed0 does not guarantee their mean.', flush=True)
    print('[CAME] Task07 is the documented main-task order, not a legacy1471 queue index.', flush=True)
    sys.argv += ['--delivery', str(root / 'artifacts/remaining_main_pilots/two_lp_priority/verified_job07_sports_delivery.json')]
    from run_verified_two_lp_repeats import main
    main()
