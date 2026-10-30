"""Numerically verify which official feature block matches a pinned CLIP text encoder."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast
from came_repro.author_loader import fingerprint
from came_repro.experiment import write_json

RAW_TEXT = {'ele-fashion': 'ele-fashion', 'sports-copurchase': 'sports', 'cloth-copurchase': 'cloth', 'books-lp': 'books-lp', 'books-nc': 'books-nc'}
NODE_COUNTS = {'ele-fashion': 97766, 'sports-copurchase': 50250, 'cloth-copurchase': 125839, 'books-lp': 636502, 'books-nc': 685294}


def verified_file(directory, name):
    manifest = json.loads((directory / 'manifest.json').read_text())
    entry = manifest['files'].get(name)
    if not entry or fingerprint(directory / name) != entry['sha256']:
        raise RuntimeError(f'Unverified or changed asset: {directory / name}')
    return entry['sha256']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(RAW_TEXT), default=['ele-fashion', 'sports-copurchase', 'cloth-copurchase'])
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.samples < 8:
        raise ValueError('CUDA and at least 8 samples are required')
    torch.set_num_threads(4)
    model_dir = ROOT / 'data/models/clip_vit_base_patch32'
    model_manifest = json.loads((model_dir / 'manifest.json').read_text())
    for name in model_manifest['files']:
        verified_file(model_dir, name)
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True).cuda().eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    result = {'model': model_manifest['repository'], 'model_revision': model_manifest['revision'],
              'model_weights_sha256': model_manifest['files']['pytorch_model.bin']['sha256'],
              'precision': 'fp32', 'device': torch.cuda.get_device_name(0),
              'method': 'seeded random node IDs, raw ID mapping, normalized text embedding against both 512D feature blocks',
              'datasets': {}}
    output = ROOT / 'artifacts/audit/clip_text_verification.json'
    for dataset in args.datasets:
        raw_root = ROOT / 'data/raw/mm_graph'
        text_root = ROOT / 'data/raw/mm_graph_text'
        text_name = f'raw-data/{RAW_TEXT[dataset]}-raw-text.jsonl'
        hashes = {'raw_text': verified_file(text_root, text_name),
                  'node_mapping': verified_file(raw_root, dataset + '/node_mapping.pt'),
                  'clip_features': verified_file(raw_root, dataset + '/clip_feat.pt')}
        mapping = torch.load(raw_root / dataset / 'node_mapping.pt', map_location='cpu', weights_only=False)
        n = NODE_COUNTS[dataset]
        mapping_compatible = len(mapping) == n and set(int(index) for index in mapping.values()) == set(range(n))
        inverse = {int(index): str(identifier) for identifier, index in mapping.items()} if mapping_compatible else {}
        chosen = sorted(np.random.default_rng(20261010).choice(n, size=min(args.samples, n), replace=False).tolist())
        wanted = {inverse[index]: index for index in chosen} if mapping_compatible else {}
        alignment_method = 'official_raw_id_to_node_mapping' if mapping_compatible else 'raw_text_row_order_candidate_verified_by_feature_match'
        candidate_rows = set(chosen)
        aligned = {}
        row_count = 0
        with (text_root / text_name).open(encoding='utf-8') as stream:
            for row_index, line in enumerate(stream):
                row_count += 1
                item = json.loads(line)
                identifier = str(item.get('asin', item.get('book_id', item.get('id', ''))))
                if not mapping_compatible and row_index in candidate_rows:
                    inverse[row_index] = identifier
                    wanted[identifier] = row_index
                if identifier in wanted:
                    if identifier in aligned:
                        raise ValueError(f'Duplicate sampled raw ID: {identifier}')
                    value = item['raw_text']
                    if isinstance(value, list):
                        if not all(isinstance(part, str) for part in value):
                            raise ValueError('Raw text list contains non-string values')
                        value = ' '.join(value)
                    if not isinstance(value, str):
                        raise ValueError('Raw text must be a string or list of strings')
                    aligned[identifier] = value
        if not mapping_compatible and row_count != n:
            raise ValueError(f'Incompatible mapping and raw text row count: {row_count} != {n}')
        missing = [identifier for identifier in wanted if identifier not in aligned]
        if missing:
            raise ValueError(f'No raw text for {len(missing)} sampled node IDs: {missing[:3]}')
        texts = [aligned[inverse[index]] for index in chosen]
        with torch.no_grad():
            inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to('cuda')
            recomputed = F.normalize(model.get_text_features(**inputs), dim=-1).cpu()
        try:
            features = torch.load(str(raw_root / dataset / 'clip_feat.pt'), map_location='cpu', weights_only=True, mmap=True)
        except (RuntimeError, ValueError):
            features = torch.load(raw_root / dataset / 'clip_feat.pt', map_location='cpu', weights_only=True)
        sampled = features[chosen].float()
        del features
        matches = []
        for block in (sampled[:, :512], sampled[:, 512:]):
            cosine = F.cosine_similarity(recomputed, block).double()
            error = (recomputed - block).abs()
            matches.append({'cosine_mean': cosine.mean().item(), 'cosine_min': cosine.min().item(),
                            'max_absolute_error': error.max().item(), 'mean_absolute_error': error.mean().item(),
                            'per_node_cosine': cosine.tolist()})
        best = max(range(2), key=lambda i: matches[i]['cosine_mean'])
        verified = matches[best]['cosine_min'] >= .9999 and matches[best]['mean_absolute_error'] < .0001
        result['datasets'][dataset] = {'sha256': hashes, 'sample_node_ids': chosen, 'sample_raw_ids': [inverse[i] for i in chosen],
                                      'alignment_method': alignment_method, 'official_mapping_entries': len(mapping), 'raw_text_rows': row_count,
                                      'blocks': matches, 'text_block_index': best if verified else None,
                                      'text_block_verified': verified,
                                      'proposed_layout': ('text_first' if best == 0 else 'image_first') if verified else 'unresolved',
                                      'vision_encoder_verified': False}
        write_json(output, result)
        print(dataset, json.dumps({'verified': verified, 'best_text_block': best, 'matches': [{k:v for k,v in m.items() if k!='per_node_cosine'} for m in matches]}), flush=True)
    if not all(value['text_block_verified'] for value in result['datasets'].values()):
        raise RuntimeError('Candidate did not match all sampled datasets; do not change feature layout claims.')


if __name__ == '__main__':
    main()
