"""Independent visual-feature check using bounded samples from pinned official image archives."""
import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor
from came_repro.author_loader import fingerprint
from came_repro.experiment import write_json

ARCHIVES = {'ele-fashion': 'ele-fashion-images.tar', 'sports-copurchase': 'sports-images.tar', 'cloth-copurchase': 'cloth-images.tar',
            'books-lp': 'books-lp-images.tar', 'books-nc': 'books-nc-images.tar'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(ARCHIVES), default=list(ARCHIVES))
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    model_dir = ROOT / 'data/models/clip_vit_base_patch32'
    model_manifest = json.loads((model_dir / 'manifest.json').read_text())
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True).cuda().eval()
    processor = CLIPImageProcessor.from_pretrained(model_dir, local_files_only=True)
    text_verified = json.loads((ROOT / 'artifacts/audit/clip_text_verification.json').read_text())
    revision = json.loads((ROOT / 'data/raw/mm_graph/manifest.json').read_text())['revision']
    tree = json.loads((ROOT / 'artifacts/audit/dataset_remote_tree.json').read_text())
    image_items = {item['path']: item for item in tree if item['type'] == 'file'}
    report = {'model_revision': model_manifest['revision'], 'model_weights_sha256': model_manifest['files']['pytorch_model.bin']['sha256'],
              'method': 'First matching samples in a bounded byte range of each pinned image tar; no label selection',
              'datasets': {}}
    for dataset in args.datasets:
        mapping = torch.load(ROOT / 'data/raw/mm_graph' / dataset / 'node_mapping.pt', weights_only=False)
        if dataset == 'books-lp':
            mapping = {}
            path = ROOT / 'data/raw/mm_graph_text/raw-data/books-lp-raw-text.jsonl'
            with path.open(encoding='utf-8') as stream:
                for node, line in enumerate(stream):
                    item_text = json.loads(line)
                    identifier = str(item_text['asin'])
                    if identifier in mapping:
                        raise ValueError('Duplicate book raw ID')
                    mapping[identifier] = node
            if len(mapping) != 636502 or not text_verified['datasets'][dataset]['text_block_verified']:
                raise ValueError('Goodreads-LP raw row alignment has not passed independent text checks')
        else:
            mapping = {str(key): int(value) for key, value in mapping.items()}
        filename = 'raw-data/' + ARCHIVES[dataset]
        item = image_items[filename]
        count = min(16 * 1024 * 1024, item['size'])
        directory = ROOT / 'data/raw/mm_graph_image_samples' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / 'archive_prefix.bin'
        if not prefix.exists():
            url = f'https://huggingface.co/datasets/mm-graph-org/mm-graph/resolve/{revision}/{filename}?sample_prefix=16MiB'
            request = urllib.request.Request(url, headers={'Range': f'bytes=0-{count-1}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers['Content-Range'] != f'bytes 0-{count-1}/{item["size"]}':
                    raise RuntimeError('Image archive sample range not honored')
                raw = response.read(count + 1)
            if len(raw) != count:
                raise RuntimeError('Truncated archive sample')
            prefix.write_bytes(raw)
        raw = prefix.read_bytes()
        images, nodes, evidence = [], [], []
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r|*') as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                identifier = Path(member.name).stem
                if identifier not in mapping:
                    continue
                payload = archive.extractfile(member).read()
                images.append(Image.open(io.BytesIO(payload)).convert('RGB'))
                nodes.append(mapping[identifier])
                (directory / (identifier + Path(member.name).suffix)).write_bytes(payload)
                evidence.append({'node_id': mapping[identifier], 'raw_id': identifier, 'archive_member': member.name,
                                 'image_sha256': hashlib.sha256(payload).hexdigest()})
                if len(images) >= args.samples:
                    break
        if len(images) < args.samples:
            raise ValueError(f'Insufficient matching images in prefix: {dataset} {len(images)}')
        with torch.no_grad():
            if dataset.startswith('books-'):
                images = [image.resize((224, 224), Image.Resampling.BICUBIC) for image in images]
                image_inputs = processor(images=images, do_resize=False, do_center_crop=False, return_tensors='pt')
                preprocessing = 'Resize to 224x224 square with PIL bicubic, then CLIP rescale/normalize; empirically matched'
            else:
                image_inputs = processor(images=images, return_tensors='pt')
                preprocessing = 'Official CLIP image processor; sampled source images are square'
            recomputed = F.normalize(model.get_image_features(**image_inputs.to('cuda')), dim=-1).cpu()
        text_block = text_verified['datasets'][dataset]['text_block_index']
        features = torch.load(ROOT / 'data/raw/mm_graph' / dataset / 'clip_feat.pt', weights_only=True, map_location='cpu')
        vision_block = features[nodes, 512:] if text_block == 0 else features[nodes, :512]
        del features
        cosine = F.cosine_similarity(recomputed, vision_block, dim=-1)
        error = (recomputed - vision_block).abs()
        verified = cosine.min().item() > .9999 and error.mean().item() < .0001
        report['datasets'][dataset] = {'revision': revision, 'source_archive': filename, 'publisher_full_archive_sha256': item['lfs']['oid'],
              'downloaded_range': [0, count-1], 'prefix_sha256': fingerprint(prefix), 'full_archive_downloaded': False,
              'samples': evidence, 'cosine_min': cosine.min().item(), 'cosine_mean': cosine.double().mean().item(),
              'preprocessing': preprocessing,
              'per_node_cosine': cosine.tolist(), 'mean_absolute_error': error.mean().item(), 'max_absolute_error': error.max().item(),
              'vision_encoder_verified': verified}
        write_json(ROOT / 'artifacts/audit/clip_image_verification.json', report)
        print(dataset, verified, 'cosine_min=', cosine.min().item(), 'max_error=', error.max().item(), flush=True)
    if not all(item['vision_encoder_verified'] for item in report['datasets'].values()):
        raise RuntimeError('Visual candidate mismatch; do not claim verified vision backbone')


if __name__ == '__main__':
    main()
