"""Require both torch and DGL CUDA and exercise the full author architecture."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('DGLDEFAULTDIR', str(ROOT / '.runtime/dgl'))

import torch
import dgl
from came_repro.author_loader import load_author_model, verify_originals


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch required; CPU fallback forbidden')
    graph = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]), num_nodes=4).to('cuda')
    graph = dgl.add_self_loop(graph)
    layer = dgl.nn.GATConv(8, 4, 2).cuda()
    values = torch.randn(4, 8, device='cuda')
    layer(graph, values).sum().backward()
    model = load_author_model()(512, 512, hidden_1024=1024, out_1024=1024, num_experts=18,
                                w12=0.5, w13=0.1, w23=0.5, loss_batch_size=None).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = model(graph, torch.randn(4, 512, device='cuda'), torch.randn(4, 512, device='cuda'))
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite model loss')
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    report = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'torch': torch.__version__, 'dgl': dgl.__version__, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
              'parameters': sum(p.numel() for p in model.parameters()), 'loss': loss.item(),
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'torch_cuda': True, 'dgl_cuda': True, 'author_backward_optimizer': True,
              'original_files': verify_originals()}
    (ROOT / 'artifacts/audit/environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'artifacts/audit/pip_freeze.txt').write_text(lock, encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
