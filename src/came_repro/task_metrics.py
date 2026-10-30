"""Task metrics and episodic sampling with explicit benchmark semantics."""
import numpy as np
import torch
import torch.nn.functional as F


def ranking_metrics(positive, negative):
    """OGB/MM-Graph average rank for ties, not an arbitrary tie-break ordering."""
    if positive.ndim != 1 or negative.ndim != 2 or len(positive) != len(negative):
        raise ValueError('Ranking scores require (N,) positives and (N,K) negatives')
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError('Nonfinite ranking scores')
    ranks = 1 + .5 * ((negative > positive[:, None]).sum(1) + (negative >= positive[:, None]).sum(1))
    ranks = ranks.double()
    return {'mrr': 1/ranks, 'hits@1': (ranks <= 1).double(), 'hits@3': (ranks <= 3).double(), 'hits@10': (ranks <= 10).double()}


def support_query_episode(labels, candidates, way, shot, seed):
    labels, candidates = np.asarray(labels), np.asarray(candidates, dtype=np.int64)
    random = np.random.default_rng(seed)
    classes = [int(value) for value in np.unique(labels[candidates]) if value >= 0 and np.sum(labels[candidates] == value) > shot]
    if len(classes) < way:
        raise ValueError('Insufficient classes with disjoint support/query sets')
    chosen = random.choice(classes, size=way, replace=False)
    support, query = [], []
    for class_id in chosen:
        nodes = random.permutation(candidates[labels[candidates] == class_id])
        support.append(nodes[:shot])
        query.append(nodes[shot:])
    return chosen, np.stack(support), np.concatenate(query)


def prototype_predictions(embeddings, chosen_classes, support, queries, batch_size=4096):
    prototypes = torch.from_numpy(np.stack([np.asarray(embeddings[index]).mean(0) for index in support])).float().cuda()
    prototypes = F.normalize(prototypes, dim=-1)
    predictions = []
    for start in range(0, len(queries), batch_size):
        features = torch.from_numpy(np.array(embeddings[queries[start:start+batch_size]], dtype=np.float32, copy=True)).cuda()
        nearest = (F.normalize(features, dim=-1) @ prototypes.T).argmax(1).cpu().numpy()
        predictions.append(np.asarray(chosen_classes)[nearest])
    return np.concatenate(predictions)


def reverse_overlap_mask(train_source, train_target, source, target, num_nodes):
    """True queries have a node pair already present in the directed training graph."""
    train_pairs = np.unique(np.minimum(train_source, train_target)*num_nodes + np.maximum(train_source, train_target))
    queries = np.minimum(source, target)*num_nodes + np.maximum(source, target)
    return np.isin(queries, train_pairs)
