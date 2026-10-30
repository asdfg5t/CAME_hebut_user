"""Download pinned official MM-Graph assets with resumable transfers and SHA256 checks."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import builtins
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'mm-graph-org/mm-graph'
DATASETS = ['ele-fashion', 'sports-copurchase', 'cloth-copurchase', 'books-lp', 'books-nc']
_LOG_LOCK = threading.Lock()


def print(*args, **kwargs):
    message = ' '.join(str(value) for value in args)
    with _LOG_LOCK:
        builtins.print(*args, **kwargs)
        log_path = ROOT / 'artifacts/audit/dataset_download.jsonl'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'message': message}) + '\n')


def save_manifest(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ranged_download(item, revision, output, workers, asset_url=None):
    target = output / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    size = item['size']
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise RuntimeError('Partial file larger than publisher size')
    directory = target.with_suffix(target.suffix + '.ranges')
    directory.mkdir(exist_ok=True)
    chunk_size = 64 * 1024 * 1024
    tasks = [(start, min(size - 1, start + chunk_size - 1)) for start in range(offset, size, chunk_size)]

    def fetch_range(bounds):
        start, end = bounds
        chunk = directory / f'{start}-{end}.part'
        complete = directory / f'{start}-{end}.done'
        expected_size = end - start + 1
        if complete.exists() and complete.stat().st_size == expected_size:
            return complete
        for attempt in range(6):
            consumed = chunk.stat().st_size if chunk.exists() else 0
            if consumed == expected_size:
                chunk.replace(complete)
                return complete
            if consumed > expected_size:
                raise RuntimeError(f'Bad range chunk size: {chunk}')
            position = start + consumed
            base_url = asset_url or f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item["path"]}?download=true'
            url = base_url + ('&' if '?' in base_url else '?') + f'range_start={position}'
            request = urllib.request.Request(url, headers={'Range': f'bytes={position}-{end}'})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected_range = f'bytes {position}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                        raise RuntimeError(f'Range not honored: {response.status} {response.headers.get("Content-Range")}')
                    with chunk.open('ab') as stream:
                        left = expected_size - consumed
                        while left:
                            block = response.read(min(1024 * 1024, left))
                            if not block:
                                raise RuntimeError('Truncated range')
                            stream.write(block)
                            left -= len(block)
                chunk.replace(complete)
                return complete
            except Exception as error:
                print(f'Range retry {attempt + 1}/6 {item["path"]} offset={position}: {error}', flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_range, bounds) for bounds in tasks]
        completed = offset
        for future in as_completed(futures):
            completed += future.result().stat().st_size
            print(f'{item["path"]}: verified-length ranges {completed / size:.1%} ({completed / 1e6:.0f}/{size / 1e6:.0f} MB)', flush=True)
    with partial.open('ab') as stream:
        for start, end in tasks:
            chunk = directory / f'{start}-{end}.done'
            with chunk.open('rb') as source:
                shutil.copyfileobj(source, stream, 4 * 1024 * 1024)
            stream.flush()
    if partial.stat().st_size != size:
        raise RuntimeError('Assembled asset size mismatch')
    digest = sha256(partial)
    expected = item.get('lfs', {}).get('oid')
    if expected and digest != expected:
        raise RuntimeError(f'Assembled file SHA256 mismatch: {partial}')
    partial.replace(target)
    # Delete only explicit range files in this asset's project-local range directory.
    for start, end in tasks:
        (directory / f'{start}-{end}.done').unlink()
    print(f'Verified {item["path"]} sha256={digest}', flush=True)
    return digest


def download(item, revision, output, range_workers=1, asset_url=None):
    target = output / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('lfs', {}).get('oid')
    size = item['size']
    if target.exists() and target.stat().st_size == size:
        digest = sha256(target)
        if expected and digest != expected:
            raise RuntimeError(f'Existing file checksum mismatch: {target}')
        return digest
    if range_workers > 1 and size > 128 * 1024 * 1024:
        return ranged_download(item, revision, output, range_workers, asset_url)
    partial = target.with_suffix(target.suffix + '.part')
    url = asset_url or f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item["path"]}'
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Invalid resume range')
                with partial.open('ab' if resumed else 'wb') as stream:
                    count = offset if resumed else 0
                    last = time.monotonic()
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        count += len(block)
                        if time.monotonic() - last > 15:
                            print(f'{item["path"]}: {count / size:.1%} ({count / 1e6:.0f}/{size / 1e6:.0f} MB)', flush=True)
                            last = time.monotonic()
            if partial.stat().st_size != size:
                raise RuntimeError(f'Truncated transfer: {partial}')
            digest = sha256(partial)
            if expected and digest != expected:
                raise RuntimeError(f'SHA256 mismatch: {partial}; remove this partial file before retry')
            partial.replace(target)
            print(f'Verified {item["path"]} sha256={digest}', flush=True)
            return digest
        except Exception as error:
            print(f'Attempt {attempt + 1}/6 failed for {item["path"]}: {error}', flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--range-workers', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.range_workers <= 8:
        parser.error('--range-workers must be between 1 and 8')
    output = ROOT / 'data/raw/mm_graph'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = old.get('revision') or get_json(f'https://huggingface.co/api/datasets/{REPO}')['sha']
    tree = get_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}?recursive=true&expand=false')
    wanted = {'clip_feat.pt', 'split.pt', 'labels-w-missing.pt', 'nc_edges-nodeid.pt', 'lp-edge-split.pt', 'node_mapping.pt'}
    items = [item for dataset in args.datasets for item in tree
             if item['type'] == 'file' and item['path'].startswith(dataset + '/')
             and Path(item['path']).name in wanted]
    remaining = sum(max(0, i['size'] - ((output / i['path']).stat().st_size if (output / i['path']).exists() else 0)) for i in items)
    free = shutil.disk_usage(output).free
    print(f'Pinned revision: {revision}; download: {remaining / 1e9:.2f} GB; free: {free / 1e9:.2f} GB', flush=True)
    manifest = {'repository': REPO, 'revision': revision, 'source': 'https://github.com/mm-graph-benchmark/mm-graph-benchmark',
                'files': old.get('files', {}), 'selection': [i['path'] for i in items]}
    save_manifest(manifest_path, manifest)
    if args.plan_only:
        return
    assembly_reserve = max((i['size'] for i in items), default=0) if args.range_workers > 1 else 0
    if remaining + assembly_reserve + 3 * 1024 ** 3 > free:
        raise RuntimeError('Insufficient space: must retain 3 GiB reserve; select fewer datasets or move data explicitly.')
    for item in items:
        digest = download(item, revision, output, args.range_workers)
        manifest['files'][item['path']] = {'size': item['size'], 'sha256': digest,
                                         'publisher_sha256': item.get('lfs', {}).get('oid'), 'verified': True}
        save_manifest(manifest_path, manifest)


if __name__ == '__main__':
    main()
