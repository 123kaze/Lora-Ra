"""Mergeable row/column/full-matrix adapters for the expanded target experiment.

The legacy input-column implementation is preserved for old experiment identities.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AxisFactor(nn.Module):
    def __init__(self, linear, axis, indices, rank, alpha):
        super().__init__()
        if axis not in ('row', 'column'):
            raise ValueError(axis)
        limit = linear.out_features if axis == 'row' else linear.in_features
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= limit:
            raise ValueError('Invalid residual-axis indices')
        m = len(indices) if axis == 'row' else linear.out_features
        n = len(indices) if axis == 'column' else linear.in_features
        if not 1 <= rank <= min(m, n):
            raise ValueError('Invalid adapter rank')
        self.axis = axis
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.A = nn.Parameter(torch.empty(rank, n))
        self.B = nn.Parameter(torch.zeros(m, rank))
        nn.init.normal_(self.A, std=.02)
        self.scaling = float(alpha) / math.sqrt(rank)
        self.to(device=linear.weight.device, dtype=linear.weight.dtype)

    def forward(self, x):
        if self.axis == 'column':
            x = x.index_select(-1, self.indices)
        return F.linear(F.linear(x, self.A), self.B) * self.scaling


class AxisGroupedLinear(nn.Module):
    def __init__(self, base, specifications, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.factors = nn.ModuleList([AxisFactor(base, axis, indices, rank, alpha)
                                      for axis, indices, rank in specifications])

    def forward(self, x):
        output = self.base(x)
        for factor in self.factors:
            update = factor(x)
            output = (output.index_add(-1, factor.indices, update) if factor.axis == 'row'
                      else output + update)
        return output

    def merged(self):
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None).to(self.base.weight)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
            for factor in self.factors:
                delta = (factor.B @ factor.A) * factor.scaling
                merged.weight.index_add_(0 if factor.axis == 'row' else 1, factor.indices, delta)
        return merged.requires_grad_(False)


def attach_axes(model, groups, ranks, alpha=math.sqrt(3)):
    model.requires_grad_(False)
    selected = {}
    for group in groups:
        rank = int(ranks.get(group['id'], 0))
        if rank < 0:
            raise ValueError('Negative rank')
        if not rank:
            continue
        linear = model.get_submodule(group['module'])
        if not isinstance(linear, nn.Linear):
            raise TypeError('Expected an unwrapped linear module')
        axis = group.get('axis', 'column')
        indices = group.get('indices', list(range(linear.in_features)))
        m = len(indices) if axis == 'row' else linear.out_features
        n = len(indices) if axis == 'column' else linear.in_features
        if (m, n, m+n) != (group['m'], group['n'], group['cost']):
            raise ValueError('Parameter count metadata mismatch')
        selected.setdefault(group['module'], []).append((axis, indices, rank))
    for name, specifications in selected.items():
        parent, attribute = name.rsplit('.', 1)
        linear = model.get_submodule(name)
        setattr(model.get_submodule(parent), attribute, AxisGroupedLinear(linear, specifications, alpha))
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = sum(g['cost'] * int(ranks.get(g['id'], 0)) for g in groups)
    if actual != expected:
        raise ValueError('Actual trainable count differs from declared budget')
    return actual


def merge_axes(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, AxisGroupedLinear):
            parent, attribute = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attribute, module.merged())
