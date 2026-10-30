"""Execute preserved author files, removing only an unused missing import in memory."""
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_originals():
    baseline = json.loads((ROOT / 'artifacts/audit/original_files.json').read_text())
    for name, expected in baseline.items():
        if fingerprint(ROOT / name) != expected['sha256']:
            raise RuntimeError(f'Original author file changed: {name}')
    return baseline


def load_author_model():
    verify_originals()
    namespace = '_came_preserved_author'
    if namespace + '.model' in sys.modules:
        return sys.modules[namespace + '.model'].MMgraph
    package = types.ModuleType(namespace)
    package.__path__ = [str(ROOT)]
    sys.modules[namespace] = package
    spec = importlib.util.spec_from_file_location(namespace + '.fusion_modules', ROOT / 'fusion_modules.py')
    fusion = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fusion
    spec.loader.exec_module(fusion)
    source = (ROOT / 'model.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    candidates = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'unigraph2']
    if len(candidates) != 1 or [(alias.name, alias.asname) for alias in candidates[0].names] != [('MoE', None)]:
        raise RuntimeError('Author import structure differs; review adapter before execution')
    if any(isinstance(node, ast.Name) and node.id == 'MoE' for node in ast.walk(tree)):
        raise RuntimeError('Missing MoE is used: cannot safely remove its import')
    node = candidates[0]
    lines = source.splitlines(keepends=True)
    for i in range(node.lineno - 1, node.end_lineno):
        lines[i] = '\n'
    adapted = ''.join(lines)
    model = types.ModuleType(namespace + '.model')
    model.__package__ = namespace
    model.__file__ = str(ROOT / 'model.py')
    model.ADAPTER_AUDIT = {'operation': 'remove unused from .unigraph2 import MoE in memory',
                           'source_sha256': fingerprint(ROOT / 'model.py'),
                           'executed_sha256': hashlib.sha256(adapted.encode()).hexdigest()}
    sys.modules[model.__name__] = model
    exec(compile(adapted, str(ROOT / 'model.py'), 'exec'), model.__dict__)
    return model.MMgraph
