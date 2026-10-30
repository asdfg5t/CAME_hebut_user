"""One validation-only predictor-structure comparison; author LP recipe is unreleased."""
import argparse
import json
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
RECIPE_KEYS = ('head_variant', 'lr', 'batch_size', 'max_epochs', 'patience',
               'step_lr_size', 'step_lr_gamma', 'loss_reduction')


def structure_recipe(reference):
    """Keep the previous training protocol; change only the predictor architecture."""
    recipe = {key: reference[key] for key in RECIPE_KEYS}
    if reference['negative_pool_size'] != 4 or reference['seed'] != 0:
        raise ValueError('Require the completed fixed seed0 four-way reference')
    if recipe['head_variant'] != 'existing_two_layer_head_upstream_schedule':
        raise ValueError('Reference is not the declared one-hidden-layer control')
    recipe['head_variant'] = 'two_hidden_layernorm_head_upstream_schedule'
    return recipe


def acceptance(profile, profile_path):
    import numpy as np
    import torch
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json, read_json_file
    from came_repro.experiment import Experiment, seed_everything
    from came_repro.lp_validation_repair import TrainOnlyNegatives, make_head
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from run_lp_hard_negative_candidate import train_hard, sample_hard_negative
    import came_repro.pause_control as pauses
    seed_everything(221)
    gpu = torch.randn(12, 1024, device='cuda')
    source = np.arange(6, dtype=np.int64)
    target = source + 1
    sampler = TrainOnlyNegatives(np.concatenate((source, target)), np.concatenate((target, source)), 12)
    head = make_head(profile['candidate_head_variant']).cuda()
    with use_cached_vectors():
        chosen, pool, scores = sample_hard_negative(head, gpu, source, sampler, np.random.default_rng(1))
    np.testing.assert_array_equal(chosen, pool[np.arange(6), scores.argmax(1).cpu().numpy()])
    assert not np.isin(np.repeat(source, 4) * 12 + pool.ravel(), sampler.keys).any()
    assert not (pool == source[:, None]).any()
    train = {'source_node': source, 'target_node': target}
    valid = {'source_node': np.array([6, 7]), 'target_node': np.array([7, 8]),
             'target_node_neg': np.tile(np.arange(12), (2, 13))[:, :150]}
    recipe = dict(profile['matched_recipe'], head_variant=profile['candidate_head_variant'],
                  max_epochs=3, batch_size=3)
    config = {'name': 'cloth_head_resume_fixture', 'seed': 1,
              'storage': read_json_file(ROOT / 'configs/storage.json'),
              'evaluation': {'fixture_only_not_benchmark': True, 'head_variant': recipe['head_variant']}}
    baseline = Experiment(config, {'fixture_only': True})
    try:
        with use_cached_vectors():
            train_hard(gpu, train, valid, recipe, 1, baseline)
        baseline.status('completed_acceptance_fixture')
    finally:
        baseline.close()
    paused = Experiment(config, {'fixture_only': True})
    original_pause = pauses.pause_requested
    calls = [0]
    def pause_after_two():
        calls[0] += 1
        return calls[0] == 2
    pauses.pause_requested = pause_after_two
    try:
        try:
            with use_cached_vectors():
                train_hard(gpu, train, valid, recipe, 1, paused)
        except pauses.TrainingPaused:
            paused.status('paused_acceptance_fixture')
        else:
            raise ValueError('LayerNorm resume fixture did not pause')
    finally:
        pauses.pause_requested = original_pause
        paused.close()
    saved = paused.checkpoint_directory / 'link_head_last.pt'
    resumed = Experiment(config, {'fixture_only': True}, saved)
    try:
        with use_cached_vectors():
            train_hard(gpu, train, valid, recipe, 1, resumed, saved)
        resumed.status('completed_acceptance_fixture')
    finally:
        resumed.close()
    a = torch.load(str(baseline.checkpoint_directory / 'link_head_last.pt'), map_location='cpu', weights_only=False)
    b = torch.load(str(resumed.checkpoint_directory / 'link_head_last.pt'), map_location='cpu', weights_only=False)
    assert a['history_rows'] == b['history_rows'] and a['next_epoch'] == b['next_epoch'] == 4
    for key in a['head']:
        assert torch.equal(a['head'][key], b['head'][key])
    report = {'passed': True, 'actual_Torch_DGL_CUDA': True, 'head_variant': recipe['head_variant'],
              'hard_selection_and_known_nonedge_verified': True,
              'seed1_complete_optimizer_RNG_resume_bitexact': True,
              'fixture_not_a_benchmark': True, 'profile_sha256': fingerprint(profile_path),
              'script_sha256': fingerprint(__file__),
              'training_backend_sha256': fingerprint(ROOT / 'scripts/run_lp_hard_negative_candidate.py'),
              'parent_checkpoint_sha256': fingerprint(saved),
              'fixture_runs': [str(e.directory) for e in (baseline, paused, resumed)]}
    atomic_write_json(ROOT / 'artifacts/audit/cloth_predictor_CUDA_acceptance_20261015.json', report)
    print(json.dumps(report))


def main():
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.experiment import Experiment
    from came_repro.protocol_repair import check_cuda
    from came_repro.lp_validation_repair import load_verified_cache
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.pause_control import TrainingPaused
    from run_lp_hard_negative_candidate import train_hard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', action='store_true')
    parser.add_argument('--resume-head', type=Path)
    args = parser.parse_args()
    verify_originals()
    check_cuda()
    torch.set_num_threads(4)
    profile_path = ROOT / 'configs/cloth_predictor_structure_20261015.json'
    profile = read_json_file(profile_path)
    if args.acceptance:
        acceptance(profile, profile_path)
        return
    accepted = read_json_file(ROOT / 'artifacts/audit/cloth_predictor_CUDA_acceptance_20261015.json')
    if (not accepted['passed'] or accepted['profile_sha256'] != fingerprint(profile_path)
        or accepted['script_sha256'] != fingerprint(__file__)
        or accepted['training_backend_sha256'] != fingerprint(ROOT / 'scripts/run_lp_hard_negative_candidate.py')):
        raise ValueError('Exact predictor-structure CUDA/resume acceptance required')
    reference = ROOT / profile['reference_validation_run']
    if fingerprint(reference / 'summary.json') != profile['reference_validation_summary_sha256']:
        raise ValueError('Fixed validation reference changed')
    summary0 = read_json_file(reference / 'summary.json')
    recipe = structure_recipe(summary0)
    if recipe != dict(profile['matched_recipe'], head_variant=profile['candidate_head_variant']):
        raise ValueError('Matched experiment changed more than architecture')
    digest = summary0['encoder_sha256']
    if digest != profile['encoder_sha256']:
        raise ValueError('Protected encoder changed')
    cache = Path(summary0['embedding_cache'])
    gpu, audit = load_verified_cache('cloth-copurchase', cache, digest)
    if audit.get('diagnostic_topology', {}).get('policy') != 'upstream_train_bidirected_unique':
        raise ValueError('Exact train-only reference graph required')
    config = read_json_file(reference / 'config.json')
    config.update(name='cloth_head_structure', seed=0)
    config['evaluation'].update(recipe, seed=0, predictor_profile_sha256=fingerprint(profile_path),
                                predictor_entry_sha256=fingerprint(__file__), test_used_for_selection=False)
    encoder = Path(read_json_file(ROOT / 'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(encoder) != digest:
        raise ValueError('Original shared encoder file changed')
    experiment = Experiment(config, audit, args.resume_head or encoder)
    try:
        raw = ROOT / 'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
        if fingerprint(raw) != audit['raw_sha256']['lp-edge-split.pt']:
            raise ValueError('Official split changed')
        split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
        with use_cached_vectors():
            summary = train_hard(gpu, split['train'], split['valid'], recipe, 0, experiment, args.resume_head)
        summary.update(dataset='cloth-copurchase', task='link', representation='came', encoder_sha256=digest,
                       embedding_cache=str(cache), graph_policy='upstream_train_bidirected_unique',
                       diagnostic_only=True, paper_reproduction_complete=False,
                       engineered_predictor_structure_not_author_confirmed=True)
        atomic_write_json(experiment.directory / 'summary.json', summary)
        experiment.status('completed_cloth_predictor_validation_only')
    except BaseException as error:
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error))
        raise
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
