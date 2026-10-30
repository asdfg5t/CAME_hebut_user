"""Acquire official-source assets for CLIP verification and complete paper data preparation."""
import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from download_datasets import ROOT, get_json, download, save_manifest, print


def hf_assets(kind, repository, revision, selected, output):
    api_kind = 'models' if kind == 'model' else 'datasets'
    if revision is None:
        revision = get_json(f'https://huggingface.co/api/{api_kind}/{repository}')['sha']
    tree = get_json(f'https://huggingface.co/api/{api_kind}/{repository}/tree/{revision}?recursive=true&expand=false')
    entries = {item['path']: item for item in tree if item['type'] == 'file'}
    manifest_path = output / 'manifest.json'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'repository': repository, 'revision': revision, 'files': {}}
    if manifest['revision'] != revision:
        raise RuntimeError('Existing asset directory uses another pinned revision')
    missing = [name for name in selected if name not in entries]
    if missing:
        raise ValueError(f'Assets absent from repository: {missing}')
    required = sum(entries[name]['size'] for name in selected if not (output / name).exists())
    if required * 2 + 3 * 1024**3 > shutil.disk_usage(output).free:
        raise RuntimeError('Insufficient space for transfer/assembly reserve')
    save_manifest(manifest_path, manifest)
    for name in selected:
        item = entries[name]
        prefix = '' if kind == 'model' else 'datasets/'
        url = f'https://huggingface.co/{prefix}{repository}/resolve/{revision}/{name}'
        digest = download(item, revision, output, 4, url)
        manifest['files'][name] = {'size': item['size'], 'sha256': digest, 'publisher_sha256': item.get('lfs', {}).get('oid'), 'url': url}
        save_manifest(manifest_path, manifest)
    return manifest


def direct_asset(url, name, output, workers=1):
    output.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status == 206:
            size = int(response.headers['Content-Range'].split('/')[-1])
        else:
            size = int(response.headers['Content-Length'])
        etag = response.headers.get('ETag')
    if size * 2 + 3 * 1024**3 > shutil.disk_usage(output).free:
        raise RuntimeError('Insufficient space for official archive and assembly')
    item = {'path': name, 'size': size}
    digest = download(item, 'official-url', output, workers, url)
    path = output / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {'files': {}}
    manifest['files'][name] = {'url': url, 'size': size, 'sha256': digest, 'etag': etag,
                               'publisher_sha256': None, 'integrity': 'Local SHA256 + official source URL; publisher checksum unavailable'}
    save_manifest(path, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['clip', 'mm_text', 'ogb', 'arxiv_text'], required=True)
    args = parser.parse_args()
    if args.group == 'clip':
        hf_assets('model', 'openai/clip-vit-base-patch32', '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                  ['config.json', 'merges.txt', 'preprocessor_config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'],
                  ROOT / 'data/models/clip_vit_base_patch32')
    elif args.group == 'mm_text':
        revision = json.loads((ROOT / 'data/raw/mm_graph/manifest.json').read_text())['revision']
        hf_assets('dataset', 'mm-graph-org/mm-graph', revision,
                  ['raw-data/ele-fashion-raw-text.jsonl', 'raw-data/sports-raw-text.jsonl', 'raw-data/cloth-raw-text.jsonl',
                   'raw-data/books-lp-raw-text.jsonl', 'raw-data/books-nc-raw-text.jsonl'],
                  ROOT / 'data/raw/mm_graph_text')
    elif args.group == 'ogb':
        for name in ('arxiv', 'products'):
            direct_asset(f'https://snap.stanford.edu/ogb/data/nodeproppred/{name}.zip', f'{name}.zip', ROOT / 'data/raw/ogb_archives', workers=4)
    elif args.group == 'arxiv_text':
        direct_asset('https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz', 'titleabs.tsv.gz', ROOT / 'data/raw/ogb_text')


if __name__ == '__main__':
    main()
