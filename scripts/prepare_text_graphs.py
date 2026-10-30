"""Prepare complete OGB/Wiki-CS/OFA KG inputs and resumable CUDA CLIP text features."""
import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import zipfile
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast
from came_repro.author_loader import fingerprint
from came_repro.experiment import write_json

RAW = ROOT / 'data/raw/text_graphs'
PROCESSED = ROOT / 'data/processed/text_graphs'
EXPECTED = {'ogbn-arxiv': 169343, 'ogbn-products': 2449029, 'wiki-cs': 11701, 'FB15K237': 14541, 'WN18RR': 40943}


def extract_members(archive, selection, destination):
    """Copy explicit files, checking every resolved target stays below destination."""
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(archive) as source:
        for name, relative in selection:
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive member target')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with source.open(name) as input_stream, target.open('wb') as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, 4 * 1024 * 1024)
            records[str(relative)] = {'sha256': fingerprint(target), 'size': target.stat().st_size, 'archive_member': name}
    write_json(destination / 'source_manifest.json', {'archive': str(archive.relative_to(ROOT)), 'archive_sha256': fingerprint(archive), 'files': records})


def acquire_sources():
    for short in ('arxiv', 'products'):
        archive = ROOT / 'data/raw/ogb_archives' / (short + '.zip')
        if not archive.exists():
            raise FileNotFoundError(f'Official archive is not ready: {archive}')
        with zipfile.ZipFile(archive) as source:
            selected = [(name, name) for name in source.namelist()
                        if not name.endswith('/') and ('/mapping/' in name or '/split/' in name or '/raw/' in name)
                        and not name.endswith('node-feat.csv.gz')]
        extract_members(archive, selected, RAW / ('ogbn-' + short))
    archive = next((ROOT / 'data/raw/source_archives').glob('wikics-*.zip'))
    with zipfile.ZipFile(archive) as source:
        selected = [(name, Path(name).name) for name in source.namelist() if '/dataset/' in name and Path(name).name in ('data.json', 'metadata.json', 'statistics.json')]
    extract_members(archive, selected, RAW / 'wiki-cs')
    archive = next((ROOT / 'data/raw/source_archives').glob('ofa-*.zip'))
    for dataset in ('FB15K237', 'WN18RR'):
        with zipfile.ZipFile(archive) as source:
            selected = [(name, Path(name).name) for name in source.namelist() if f'/data/KG/{dataset}/' in name and not name.endswith('/')]
        extract_members(archive, selected, RAW / dataset)


def save_inputs(dataset, edge_array, labels, splits, texts, metadata):
    directory = PROCESSED / dataset
    directory.mkdir(parents=True, exist_ok=True)
    if len(texts) != EXPECTED[dataset]:
        raise ValueError(f'Full node count mismatch: {dataset} {len(texts)} vs {EXPECTED[dataset]}')
    np.save(directory / 'edge_index.npy', np.asarray(edge_array, dtype=np.int64))
    if labels is not None:
        np.save(directory / 'labels.npy', np.asarray(labels, dtype=np.int64))
    for name, values in splits.items():
        np.save(directory / (name + '_indices.npy'), np.asarray(values, dtype=np.int64))
    text_path = directory / 'node_text.jsonl'
    with text_path.open('w', encoding='utf-8') as stream:
        for node, text in enumerate(texts):
            stream.write(json.dumps({'node_id': node, 'text': text}, ensure_ascii=False) + '\n')
    manifest = {'dataset': dataset, 'nodes': len(texts), 'edges': len(edge_array), 'raw_source': json.loads((RAW / dataset / 'source_manifest.json').read_text()),
                'text_sha256': fingerprint(text_path), 'text_policy': metadata, 'feature_policy': 'CLIP text reused as visual placeholder per paper section 5.2',
                'files': {path.name: fingerprint(path) for path in directory.glob('*.npy')}}
    write_json(directory / 'manifest.json', manifest)


def prepare_wiki():
    dataset = 'wiki-cs'
    raw = json.loads((RAW / dataset / 'data.json').read_text())
    meta = json.loads((RAW / dataset / 'metadata.json').read_text())
    edges = [(node, neighbor) for node, links in enumerate(raw['links']) for neighbor in links]
    texts = [(node['title'] + '. ' + ' '.join(node['tokens'])).strip() for node in meta['nodes']]
    # Keep all 20 official masks, including stopping masks, rather than rebuilding splits.
    splits = {'train_masks': np.asarray(raw['train_masks'], dtype=np.int64), 'val_masks': np.asarray(raw['val_masks'], dtype=np.int64),
              'stopping_masks': np.asarray(raw['stopping_masks'], dtype=np.int64), 'test_mask': np.asarray(raw['test_mask'], dtype=np.int64)}
    save_inputs(dataset, edges, raw['labels'], splits, texts, 'Official Wiki-CS title + tokens; no task/label text added. Directed original links and 20 official splits retained.')


def prepare_kg(dataset):
    raw = RAW / dataset
    if dataset == 'WN18RR':
        entries = [line.rstrip('\n').split('\t', 1) for line in (raw / 'entity2text.txt').read_text(encoding='utf-8').splitlines()]
        entities, texts = [row[0] for row in entries], [row[1] for row in entries]
    else:
        entries = json.loads((raw / 'entity2wikidata.json').read_text())
        entities = list(entries)
        texts = [f'entity names: {entries[key]["label"]}, entity alternatives: ' + ', '.join(entries[key]['alternatives']) + '. entity descriptions:' + (entries[key]['description'] or '') for key in entities]
    entity_ids = {entity: i for i, entity in enumerate(entities)}
    relation_ids, triples = {}, {}
    for split in ('train', 'valid', 'test'):
        rows = []
        for line in (raw / (split + '.txt')).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            head, relation, tail = line.split()
            for entity in (head, tail):
                if entity not in entity_ids:
                    entity_ids[entity] = len(texts)
                    entities.append(entity)
                    texts.append('entity names: Unknown')
            relation_ids.setdefault(relation, len(relation_ids))
            rows.append([entity_ids[head], relation_ids[relation], entity_ids[tail]])
        triples[split] = np.asarray(rows, dtype=np.int64)
    save_inputs(dataset, triples['train'][:, [0, 2]], None, {}, texts, 'OFA entity ordering and textual metadata; no relation text in node features. Train triples only in topology.')
    directory = PROCESSED / dataset
    for split, value in triples.items():
        np.save(directory / (split + '_triples.npy'), value)
    write_json(directory / 'entity_relation_mapping.json', {'entities': entities, 'relations': relation_ids})
    manifest = json.loads((directory / 'manifest.json').read_text())
    manifest['relations'] = len(relation_ids)
    manifest['files'].update({path.name: fingerprint(path) for path in directory.glob('*triples.npy')})
    write_json(directory / 'manifest.json', manifest)


def prepare_ogb(dataset):
    short = dataset.removeprefix('ogbn-')
    raw = RAW / dataset / short
    edge = pd.read_csv(raw / 'raw/edge.csv.gz', header=None, dtype=np.int64).values
    labels = pd.read_csv(raw / 'raw/node-label.csv.gz', header=None, dtype=np.int64).values.flatten()
    splits = {name.replace('valid', 'val'): pd.read_csv(raw / ('split/time' if short == 'arxiv' else 'split/sales_ranking') / (name + '.csv.gz'), header=None, dtype=np.int64).values.flatten()
              for name in ('train', 'valid', 'test')}
    if short == 'arxiv':
        mapping = pd.read_csv(raw / 'mapping/nodeidx2paperid.csv.gz')
        titleabs = pd.read_csv(ROOT / 'data/raw/ogb_text/titleabs.tsv.gz', sep='\t', names=['paper id', 'title', 'abstract'], dtype=str)
        indexed = titleabs.drop_duplicates('paper id').set_index('paper id')
        texts = []
        missing = 0
        for identifier in mapping['paper id'].astype(str):
            if identifier not in indexed.index:
                missing += 1
                texts.append('')
            else:
                row = indexed.loc[identifier]
                texts.append((str(row['title']) + '. ' + str(row['abstract'])).strip())
        if missing:
            raise ValueError(f'Arxiv texts missing for {missing} nodes')
    else:
        mapping = pd.read_csv(raw / 'mapping/nodeidx2asin.csv.gz', dtype=str)
        ids = {identifier: int(index) for index, identifier in zip(mapping['node idx'], mapping['asin'])}
        texts = [None] * EXPECTED[dataset]
        archive = ROOT / 'data/raw/ogb_text/Amazon-3M.raw.zip'
        with zipfile.ZipFile(archive) as source:
            selected = [name for name in source.namelist() if name.endswith(('trn.json.gz', 'tst.json.gz'))]
            if len(selected) != 2:
                raise ValueError('Unexpected official Amazon-3M text archive structure')
            for name in selected:
                with source.open(name) as compressed, gzip.open(compressed, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        item = json.loads(line)
                        if item['uid'] in ids:
                            node = ids[item['uid']]
                            text = (item.get('title', '') + '. ' + item.get('content', '')).strip()
                            if texts[node] is not None and texts[node] != text:
                                raise ValueError('Conflicting raw text for a Products node')
                            texts[node] = text
        missing = sum(value is None for value in texts)
        if missing:
            raise ValueError(f'Products texts missing for {missing} nodes; no dummy substitution allowed')
    # OGB documents Products as undirected and requires inverse edges; retain both directions.
    if short == 'products':
        edge = np.concatenate([edge, edge[:, ::-1]], axis=0)
    save_inputs(dataset, edge, labels, splits, texts, 'Official node ID mapping; title + description/abstract. CLIP truncates to 77 tokens; no class text added.')


def encode(dataset, batch_size, max_nodes=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for CLIP encoding')
    directory = PROCESSED / dataset
    manifest = json.loads((directory / 'manifest.json').read_text())
    model_dir = ROOT / 'data/models/clip_vit_base_patch32'
    model_manifest = json.loads((model_dir / 'manifest.json').read_text())
    identity = {'text_sha256': manifest['text_sha256'], 'model_revision': model_manifest['revision'],
                'model_weights_sha256': model_manifest['files']['pytorch_model.bin']['sha256'], 'normalization': 'L2', 'max_length': 77}
    state_path = directory / 'clip_encoding_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'identity': identity, 'cursor': 0, 'complete': False}
    if state['identity'] != identity or fingerprint(directory / 'node_text.jsonl') != identity['text_sha256']:
        raise ValueError('Encoding inputs changed; cannot resume')
    target = directory / 'clip_text.npy'
    output = np.lib.format.open_memmap(target, mode='r+' if target.exists() else 'w+', dtype=np.float32, shape=(manifest['nodes'], 512))
    state.setdefault('execution_segments', []).append({'start_node': state['cursor'], 'end_node': state['cursor'],
                 'batch_size': batch_size, 'precision': 'fp32', 'torch': torch.__version__, 'cuda': torch.version.cuda,
                 'gpu': torch.cuda.get_device_name(0), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True).cuda().eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    pending, pending_ids = [], []
    limit = min(max_nodes, manifest['nodes']) if max_nodes is not None else manifest['nodes']

    def flush():
        if not pending:
            return
        with torch.no_grad():
            inputs = tokenizer(pending, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to('cuda')
            features = F.normalize(model.get_text_features(**inputs), dim=-1).cpu().numpy()
        output[pending_ids] = features
        output.flush()
        state['cursor'] = pending_ids[-1] + 1
        state['execution_segments'][-1]['end_node'] = state['cursor']
        write_json(state_path, state)
        if state['cursor'] % (batch_size * 50) == 0:
            print(f'{dataset}: CUDA CLIP {state["cursor"]}/{manifest["nodes"]}', flush=True)
        pending.clear()
        pending_ids.clear()

    with (directory / 'node_text.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            node = item['node_id']
            if node < state['cursor']:
                continue
            if node >= limit:
                break
            pending.append(item['text'])
            pending_ids.append(node)
            if len(pending) == batch_size:
                flush()
        flush()
    state['complete'] = state['cursor'] == manifest['nodes']
    write_json(state_path, state)
    if state['complete']:
        manifest['clip_encoder'] = identity
        manifest['files']['clip_text.npy'] = fingerprint(target)
        write_json(directory / 'manifest.json', manifest)
    print(json.dumps({'dataset': dataset, **state}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['extract', 'prepare', 'encode'], required=True)
    parser.add_argument('--datasets', nargs='+', choices=list(EXPECTED), default=list(EXPECTED))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-nodes', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'extract':
        acquire_sources()
    elif args.action == 'prepare':
        for name in args.datasets:
            if name == 'wiki-cs':
                prepare_wiki()
            elif name in ('FB15K237', 'WN18RR'):
                prepare_kg(name)
            else:
                prepare_ogb(name)
            print(f'Prepared full {name}', flush=True)
    else:
        for name in args.datasets:
            encode(name, args.batch_size, args.max_nodes)


if __name__ == '__main__':
    main()
