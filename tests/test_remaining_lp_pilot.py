import numpy as np
import pytest
import torch
from came_repro import link_evaluation as link
from came_repro.remaining_lp_pilot import cached_vectors, use_cached_vectors


def test_cuda_cache_adapter_restores_streamed_vectors_on_error():
    original = link.vectors
    with pytest.raises(RuntimeError):
        with use_cached_vectors():
            assert link.vectors is cached_vectors
            raise RuntimeError('interrupted scoring')
    assert link.vectors is original


def test_cuda_cache_never_silently_accepts_cpu_or_numpy_compute():
    for matrix in (torch.ones(2,3), np.ones((2,3),dtype=np.float32)):
        with pytest.raises(ValueError):cached_vectors(matrix,np.array([0]))
