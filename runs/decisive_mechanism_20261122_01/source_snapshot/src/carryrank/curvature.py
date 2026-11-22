"""Relative-damped diagonal Kronecker proxy; not an exact Hessian/Fisher."""
from __future__ import annotations

import torch


def relative_diagonal(moment, damping=0.01, reference_mean=None):
    if moment.ndim != 1 or not torch.isfinite(moment).all() or (moment < 0).any():
        raise ValueError('Expected a finite nonnegative second-moment vector')
    if damping < 0:
        raise ValueError('Damping must be nonnegative')
    reference = moment.mean() if reference_mean is None else torch.as_tensor(reference_mean,dtype=moment.dtype,device=moment.device)
    if not torch.isfinite(reference) or reference <= 0:
        raise ValueError('Zero or invalid curvature; no absolute fallback is applied')
    result=moment+damping*reference
    if (result<=0).any():
        raise ValueError('Damped diagonal is not invertible')
    return result


def whiten(gradient,input_moment,output_moment,input_damping=0.01,output_damping=0.01,
           input_reference_mean=None,output_reference_mean=None):
    if gradient.shape != (output_moment.numel(),input_moment.numel()):
        raise ValueError('Gradient and second-moment shapes disagree')
    if not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite gradient')
    hi=relative_diagonal(input_moment,input_damping,input_reference_mean)
    ho=relative_diagonal(output_moment,output_damping,output_reference_mean)
    return gradient/ho.sqrt()[:,None]/hi.sqrt()[None,:]


def spectral_utility(matrix,max_rank=16):
    singular=torch.linalg.svdvals(matrix)[:max_rank]
    return singular,[0.0]+(0.5*singular.square()).cumsum(0).tolist()
