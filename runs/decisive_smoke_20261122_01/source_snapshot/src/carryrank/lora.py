"""Mergeable input-column adapters. Scaling is fixed to 1 for every rank."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ColumnFactor(nn.Module):
    def __init__(self, indices, out_features, rank, scaling=1.0):
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.A = nn.Parameter(torch.empty(rank, len(indices)))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        # Constant entrywise variance, not alpha/r or rank-dependent variance.
        nn.init.normal_(self.A, std=0.02)
        self.scaling = scaling

    def forward(self, x):
        return F.linear(F.linear(x.index_select(-1, self.indices), self.A), self.B) * self.scaling


class GroupedLinear(nn.Module):
    def __init__(self, base: nn.Linear, groups, scaling=1.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.factors = nn.ModuleList([
            ColumnFactor(indices, base.out_features, rank, scaling).to(base.weight.device)
            for indices, rank in groups if rank > 0
        ])

    def forward(self, x):
        y = self.base(x)
        for factor in self.factors:
            y = y + factor(x)
        return y

    def merged(self):
        result = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None).to(self.base.weight)
        with torch.no_grad():
            result.weight.copy_(self.base.weight)
            if result.bias is not None:
                result.bias.copy_(self.base.bias)
            for factor in self.factors:
                result.weight.index_add_(1, factor.indices,
                                         (factor.B @ factor.A) * factor.scaling)
        result.requires_grad_(False)
        return result


def attach(model, groups, allocation, scaling=1.0):
    model.requires_grad_(False)
    by_module = {}
    for group in groups:
        rank = allocation.get(group["id"], 0)
        if rank:
            by_module.setdefault(group["module"], []).append((group["indices"], rank))
    for name, factors in by_module.items():
        parent_name, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, GroupedLinear(getattr(parent, attr), factors, scaling))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_all(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, GroupedLinear):
            parent_name, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attr, module.merged())
