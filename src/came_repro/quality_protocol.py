"""Predeclared, validation-only selection rules for a small quality experiment."""
import copy

QUALITY_DATASETS = ('wiki-cs', 'ogbn-arxiv', 'books-nc')


def validate_profile(profile):
    if profile['datasets'] != list(QUALITY_DATASETS):
        raise ValueError('Focus on the three deficient domains; preserve Ele-fashion')
    if profile['extra_subset_rounds'] != 1 or (profile['max_active_hours'] is not None and profile['max_active_hours'] <= 0):
        raise ValueError('This pilot adds exactly one subset round with an explicit budget')
    if profile['probe_epochs'] != 2000 or profile['probe_patience'] != 100:
        raise ValueError('Screening reduces seeds, not the established head stopping budget')
    if profile['confirmation_seeds'] != [0] or profile['screen_seed'] != 0:
        raise ValueError('Use predeclared seeds, never choose favorable test seeds')
    if not 0 <= profile['wiki_split_id'] < 20:
        raise ValueError('Invalid official Wiki-CS split')
    if profile['baseline_datasets'] != list(QUALITY_DATASETS) or profile['graph_probe_datasets'] != list(QUALITY_DATASETS):
        raise ValueError('Unexpected screening scope')
    if profile['inference_batch_size'] < 1 or profile['validation_cache_mib'] < 0 or profile['owned_embedding_limit_gib'] <= 0:
        raise ValueError('Invalid memory or storage budget')
    for key in ('min_mean_validation_gain_pp', 'max_domain_validation_regression_pp', 'min_graph_validation_gain_pp'):
        if profile[key] < 0:
            raise ValueError('Selection thresholds must be nonnegative')


def select_graph_policies(reference, graph_probes, threshold_pp):
    policies = {dataset: 'original' for dataset in QUALITY_DATASETS}
    for dataset, result in graph_probes.items():
        if result['validation_accuracy'] - reference[dataset]['validation_accuracy'] >= threshold_pp / 100:
            policies[dataset] = 'bidirected'
    return policies


def validation_gate(reference, candidate, profile):
    if set(reference) != set(QUALITY_DATASETS) or set(candidate) != set(QUALITY_DATASETS):
        raise ValueError('All three domains must finish before selecting an encoder')
    gains = {dataset: 100*(candidate[dataset]['validation_accuracy']-reference[dataset]['validation_accuracy'])
             for dataset in QUALITY_DATASETS}
    mean = sum(gains.values()) / len(gains)
    passed = mean >= profile['min_mean_validation_gain_pp'] and min(gains.values()) >= -profile['max_domain_validation_regression_pp']
    return {'passed': passed, 'mean_validation_gain_pp': mean, 'domain_validation_gains_pp': gains,
            'selection_data': 'official validation only; no test metrics read',
            'paper_reproduction_complete': False}


def extension_signature(config):
    """Only identity labels, declared budget, and notes may differ from the parent."""
    value = copy.deepcopy(config)
    value.pop('name', None)
    value.pop('deviations', None)
    value['training'].pop('epochs')
    value['training'].pop('max_steps')
    return value
