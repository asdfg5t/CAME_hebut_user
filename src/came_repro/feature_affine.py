"""Training-pair-only affine conditioning; the raw-feature classifier remains linear."""
import numpy as np


def training_pair_statistics(embeddings, pairs, batch_size=8192):
    width = embeddings.shape[1]*2
    total = np.zeros(width,dtype=np.float64)
    squared = np.zeros(width,dtype=np.float64)
    count = len(pairs)
    if not count:
        raise ValueError('Training pairs required for affine statistics')
    for start in range(0,count,batch_size):
        selected = pairs[start:start+batch_size]
        values = np.concatenate([embeddings[selected[:,0]],embeddings[selected[:,1]]],axis=1)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite frozen training features')
        total += values.sum(axis=0,dtype=np.float64)
        squared += np.square(values,dtype=np.float64).sum(axis=0)
    mean = total/count
    std = np.sqrt(np.maximum(squared/count-mean**2,0))
    # Constant channels retain a finite, invertible scale; no clipping or feature removal.
    scale = np.where(std >= 1e-6,std,1.)
    return mean.astype(np.float32),scale.astype(np.float32)


def fold_affine_linear(weight, bias, mean, scale):
    """W((x-mu)/sigma)+b == W_raw x+b_raw in real arithmetic."""
    if np.any(scale<=0) or not np.isfinite(scale).all():
        raise ValueError('Finite positive affine scales required')
    raw_weight = np.asarray(weight,dtype=np.float64)/np.asarray(scale,dtype=np.float64)[None,:]
    raw_bias = np.asarray(bias,dtype=np.float64)-raw_weight@np.asarray(mean,dtype=np.float64)
    return raw_weight,raw_bias
