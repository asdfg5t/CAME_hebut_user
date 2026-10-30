"""主任务05：FB15K237，复用已核验seed0，用户点击后补seed1-9。"""
import sys
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
    print('[CAME] 主任务05 / FB15K237；前四项不重训。',flush=True)
    print('[CAME] 已核验seed0=93.481872%，论文94.59%；复用seed0，新增9个训练seed。',flush=True)
    print('[CAME] 编号05表示你的主任务训练优先顺序，不是旧1471队列的作业索引。',flush=True)
    from run_validated_remaining_tasks import main
    main()
