"""Reproduction infrastructure; author files remain at the project root."""
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('DGLDEFAULTDIR', str(_root / '.runtime/dgl'))
