import numpy as np
from came_repro.feature_affine import training_pair_statistics,fold_affine_linear


def test_statistics_use_only_training_pairs_and_preserve_constant_channels():
    class Embeddings:
        shape=(4,2)
        def __getitem__(self,index):
            assert np.max(index)<2,'Unseen/validation nodes must not set scaling'
            return np.array([[2.,5.],[4.,5.]],dtype=np.float32)[index]
    mean,scale=training_pair_statistics(Embeddings(),np.array([[0,1],[1,0]]),batch_size=1)
    np.testing.assert_array_equal(mean,[3,5,3,5])
    np.testing.assert_array_equal(scale,[1,1,1,1])


def test_folding_preserves_linear_logits_on_original_concatenated_features():
    rng=np.random.default_rng(42);x=rng.normal(size=(12,6));w=rng.normal(size=(3,6));b=rng.normal(size=3)
    mean=rng.normal(size=6);scale=np.exp(rng.normal(size=6))
    raw_w,raw_b=fold_affine_linear(w,b,mean,scale)
    np.testing.assert_allclose((x-mean)/scale@w.T+b,x@raw_w.T+raw_b,rtol=1e-12,atol=1e-12)
