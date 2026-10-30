import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('DGLDEFAULTDIR', str(ROOT / '.runtime/dgl'))
