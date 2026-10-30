"""Read-only project/cache budget check; never delete raw data or experiment records."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from came_repro.storage_budget import check_budget


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--additional-project-gib',type=float,default=0)
    parser.add_argument('--additional-cache-gib',type=float,default=0)
    args=parser.parse_args()
    report=check_budget(int(args.additional_project_gib*2**30),int(args.additional_cache_gib*2**30))
    (ROOT/'artifacts/audit/storage_budget.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
