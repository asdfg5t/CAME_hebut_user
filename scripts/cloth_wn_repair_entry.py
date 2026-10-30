"""Audited user-run handoff for the existing finite Cloth / WN experiment.

The old pretraining is resumed exactly. A separate predeclared finite LR grid
uses validation only; formal repeats and test-driven searches are not started.
"""
import argparse
import datetime
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
LEDGER = ROOT / 'artifacts/workflow_state/representation_coverage_current.json'
PAUSE = ROOT / 'artifacts/workflow_state/representation_coverage_pause.json'
OUT = ROOT / 'artifacts/cloth_wn_training_entry'
EXPECTED_UPDATES = 20480


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def validate_checkpoint(state, config, pools):
    """Reject incomplete/mismatched state, rather than silently warm-starting."""
    import torch
    from came_repro.joint_train import signature
    required = {'schema', 'config', 'model', 'optimizer', 'rng', 'schedule', 'counts', 'step', 'data_audit'}
    if not required.issubset(state) or state['schema'] != 'came_joint_v1':
        raise ValueError('A complete joint optimizer checkpoint is required')
    if signature(state['config']) != signature(config):
        raise ValueError('Checkpoint configuration differs from the frozen experiment')
    if not {'python', 'numpy', 'torch', 'cuda'}.issubset(state['rng']) or not state['rng']['cuda']:
        raise ValueError('Complete Python / NumPy / Torch / CUDA RNG required')
    s = state['schedule']; names = config['datasets']; limits = config['sampling']['seed_limits']
    batch = config['sampling']['seed_batch_size']
    if s['seed_limits'] != limits or s['batch_size'] != batch or set(s['orders']) != set(names):
        raise ValueError('Saved sampling identity changed')
    epoch = s['epoch']; step = state['step']
    if type(step) is not int or not 1 <= step <= EXPECTED_UPDATES or not 0 <= epoch <= 8:
        raise ValueError('Invalid committed optimizer step / subset-round cursor')
    if state.get('completed_epochs') != 0 or state.get('completed_subset_rounds') != epoch:
        raise ValueError('Subset rounds must not be relabeled full graph epochs')
    steps_per_round = sum((v + batch - 1)//batch for v in limits.values())
    if steps_per_round * 8 != EXPECTED_UPDATES:
        raise ValueError('Unexpected total candidate budget')
    within = 0
    for name in names:
        n = limits[name]; cursor = s['cursors'][name]
        if not torch.equal(s['fixed_seed_pools'][name].cpu(), pools[name].cpu()):
            raise ValueError('Saved unlabeled pool changed: ' + name)
        if len(s['orders'][name]) != n or not torch.equal(
                torch.sort(s['orders'][name].cpu()).values, torch.sort(pools[name].cpu()).values):
            raise ValueError('Saved epoch order is not a permutation of the fixed pool')
        if type(cursor) is not int or not 0 <= cursor <= n or cursor % batch:
            raise ValueError('Invalid committed seed cursor')
        within += cursor//batch
    # A final completed epoch retains exhausted cursors in this trainer.
    expected = epoch*steps_per_round + (within if epoch < 8 else 0)
    if step != expected or sum(c['steps'] for c in state['counts'].values()) != step:
        raise ValueError('Optimizer step and sampling cursors disagree')
    if any(c.get('skipped', 0) != 0 for c in state['counts'].values()):
        raise ValueError('Unexpected skipped training batches require independent review')
    if not state['optimizer'].get('state') or not state['optimizer'].get('param_groups'):
        raise ValueError('Adam state is missing; model-only recovery is forbidden')
    for label, values in [('model', state['model']), ('optimizer', state['optimizer'])]:
        def check(value):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('Nonfinite saved ' + label)
            elif isinstance(value, dict):
                for child in value.values(): check(child)
            elif isinstance(value, (tuple, list)):
                for child in value: check(child)
        check(values)
    return {'committed_step': step, 'completed_subset_rounds': epoch,
            'next_optimizer_step': step+1 if step < EXPECTED_UPDATES else None,
            'remaining_optimizer_updates': EXPECTED_UPDATES-step,
            'complete_model_Adam_RNG_sampling_state': True,
            'sampling_cursors': dict(s['cursors'])}


def audit_preflight(verify_cuda=False):
    import gc
    import numpy as np
    import torch
    from came_repro.author_loader import verify_originals
    from came_repro.prepared_data import load_prepared
    from came_repro.quality_pipeline import best_resume
    from came_repro.storage_budget import check_budget
    torch.set_num_threads(4)
    ledger = read(LEDGER); identity = ledger['identity']
    verify_originals()
    for path, expected in identity['sources'].items():
        if digest(path) != expected: raise ValueError('Frozen source changed: ' + path)
    config_path = ROOT/'configs/representation_coverage_20261016.json'
    plan_path = ROOT/'configs/representation_coverage_validation_20261016.json'
    if digest(config_path) != identity['config_sha256'] or digest(plan_path) != identity['plan_sha256']:
        raise ValueError('Frozen profile changed; refuse scientific protocol migration')
    plan = read(plan_path)
    for item in plan['references'].values():
        if digest(Path(item['run'])/'summary.json') != item['summary_sha256']:
            raise ValueError('Predeclared validation reference changed')
    for job in ('CUDA_capacity', 'joint_resume_acceptance'):
        record = ledger['records'][job]
        if record['state'] != 'completed' or not record['summary']['passed']:
            raise ValueError('Actual CUDA and exact-resume acceptance must already exist')
        for p,h in record['artifact_sha256'].items():
            if digest(p) != h: raise ValueError('Acceptance artifact changed')
    automatic = ROOT/'artifacts/workflow_state/cloth_wn_autonomous_current.json'
    continued = read(automatic) if automatic.exists() else None
    record = (continued['records'].get('fresh_pretraining', {}) if continued
              else ledger['records'].get('fresh_pretraining', {}))
    parent = best_resume(record, 'last.pt')
    if parent is None: raise ValueError('No complete saved pretraining checkpoint; refuse restart from zero')
    before = digest(parent)
    saved = torch.load(str(parent), map_location='cpu', weights_only=False, mmap=True)
    config = read(config_path); config.setdefault('storage', read(ROOT/'configs/storage.json'))
    pool_dir = ROOT/'artifacts/representation_coverage_20261016/seed_pools'
    pools = {}
    for name, item in plan['pools'].items():
        p = pool_dir/(name+'.npy')
        if digest(p) != item['sha256']: raise ValueError('Fixed pool file changed')
        pools[name] = torch.from_numpy(np.load(p, allow_pickle=False))
    result = validate_checkpoint(saved, config, pools)
    data_ids = {}
    for index,name in enumerate(config['datasets'],1):
        print(f'核验完整数据 {index}/3：{name}（只读散列校验）', flush=True)
        prepared = load_prepared(name, require_verified=True, build_graph=False)
        if prepared.audit != saved['data_audit'][name]:
            raise ValueError('Official data / feature identity differs from checkpoint: ' + name)
        data_ids[name] = {'nodes': prepared.audit['nodes'],
                          'prepared_manifest_sha256': prepared.audit['prepared_manifest_sha256']}
        del prepared
    if digest(parent) != before: raise ValueError('Checkpoint changed during preflight; refuse concurrent training')
    del saved; gc.collect()
    budget = check_budget(additional_project_bytes=512*2**20, additional_cache_bytes=4*2**30)
    cuda = None
    if verify_cuda:
        import dgl
        if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA required')
        a = torch.tensor([[2.,3.]], device='cuda', requires_grad=True)
        (a.square().sum()).backward()
        torch.testing.assert_close(a.grad, torch.tensor([[4.,6.]], device='cuda'))
        graph = dgl.graph(([0,1], [1,0]), num_nodes=2, device='cuda')
        graph.ndata['x'] = torch.tensor([[2.],[5.]], device='cuda')
        graph.update_all(dgl.function.copy_u('x','m'), dgl.function.sum('m','y'))
        torch.testing.assert_close(graph.ndata['y'], torch.tensor([[5.],[2.]], device='cuda'))
        cuda = {'actual_Torch_backward': True, 'actual_DGL_message_passing': True,
                'torch': torch.__version__, 'dgl': dgl.__version__, 'gpu': torch.cuda.get_device_name(0)}
        del graph,a; torch.cuda.empty_cache()
    return dict(result, parent_checkpoint=str(parent), parent_checkpoint_sha256=before,
                frozen_sources_verified=len(identity['sources']), official_data_verified=data_ids,
                storage_budget=budget, CUDA_smoke=cuda, ledger_state_before=ledger['state'],
                ledger_sha256_before=digest(LEDGER), scientific_protocol_changed=False,
                formal_repeats_started=0, score_guaranteed=False,
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def request_pause():
    from came_repro.atomic_io import atomic_write_json
    atomic_write_json(PAUSE, {'requested_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             'reason': 'Explicit user pause via run_08_cloth_wn_repair.py --pause'})
    print('已请求安全暂停：训练在下一可提交边界保存断点后退出。', flush=True)


def clear_explicit_pause(expected_digest=None):
    if PAUSE.exists():
        if expected_digest is not None and digest(PAUSE) != expected_digest:
            raise RuntimeError('Pause changed during preflight; retain the newer user request')
        archive = OUT/'pause_history'
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        # Move only this exact pause marker. Never clear other task pauses.
        PAUSE.replace(archive/(stamp+'_'+digest(PAUSE)[:12]+'.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', '--dry-run', action='store_true')
    parser.add_argument('--verify-cuda', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.pause:
        if args.resume or args.check_only or args.verify_cuda: parser.error('--pause must be used alone')
        request_pause(); return 0
    if args.check_only and args.resume: parser.error('Check-only must not clear a user pause')
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.atomic_io import atomic_write_json
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        guard = acquire_gpu_lock(ROOT/'.runtime/cloth_wn_user_entry.lock')
    except RuntimeError:
        print('已有同一入口正在执行；本次不重复启动。', flush=True); return 0
    try:
        probes = []
        try:
            for name in ('pycharm_gpu', 'representation_coverage'):
                probes.append(acquire_gpu_lock(ROOT/'.runtime'/(name+'.lock')))
        except RuntimeError:
            print('已有训练或修复任务占用GPU锁；本次不启动第二个任务。', flush=True); return 0
        finally:
            for probe in probes: probe.close()
        pause_at_start = digest(PAUSE) if PAUSE.exists() else None
        if pause_at_start is not None and not args.resume:
            print('检测到本流程主动暂停标记。确认继续时在运行参数填 --resume；没有清除暂停。', flush=True)
            return 0
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        logger = logging.getLogger('cloth_wn_user_entry'); logger.setLevel(logging.INFO)
        handler = logging.FileHandler(OUT/('entry_'+stamp+'.log'), encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
        import time
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        try:
            print('正在核验断点、原作者文件、冻结配置和完整数据身份；请等待。', flush=True)
            report = audit_preflight(args.verify_cuda)
            report['entry_sha256'] = digest(ROOT/'scripts/run_08_cloth_wn_repair.py')
            report['adapter_sha256'] = digest(__file__)
            atomic_write_json(OUT/('preflight_'+stamp+'.json'), report)
            atomic_write_json(OUT/'latest_preflight.json', report)
            logger.info(json.dumps(report, ensure_ascii=False))
            print(f"核验通过：断点step={report['committed_step']}/{EXPECTED_UPDATES}；剩余更新{report['remaining_optimizer_updates']}。", flush=True)
            print('顺序：断点续训→两项完整编码→各3个固定策略完整验证→按验证选最优→各至多一次测试。', flush=True)
            print('保留原六项结果；不自动启动十seed正式重复；不保证距论文2个百分点。', flush=True)
            if args.check_only:
                print('CHECK_ONLY_PASSED：未启动训练或模型评测。', flush=True); return 0
            if args.resume and pause_at_start is not None: clear_explicit_pause(pause_at_start)
            from cloth_wn_autonomous import main as continue_diagnostic, LEDGER as autonomous_ledger
            continue_diagnostic()
            completed = read(autonomous_ledger)
            for task in ('cloth', 'wn'):
                if task in completed.get('results', {}):
                    print(task+': '+json.dumps(completed['results'][task], ensure_ascii=False), flush=True)
                else:
                    print(task+': 验证未过预声明门槛，保留原完整测试，不新增测试。', flush=True)
            print('有限诊断结束。新结果仍需独立核验，不能直接称为达标正式重复交付。', flush=True)
            return 0
        except BaseException:
            logger.exception('入口核验或原冻结诊断停止；未修改科学协议。')
            raise
        finally:
            handler.close(); logger.removeHandler(handler)
    finally: guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
