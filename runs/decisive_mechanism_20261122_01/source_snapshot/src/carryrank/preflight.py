"""Correct residual axes and integer-budget checks for the decisive experiment."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .allocation import allocate
from .model import structure

TARGETS = [('attn.q_proj', 'column'), ('attn.k_proj', 'column'),
           ('attn.v_proj', 'column'), ('attn.o_proj', 'row'),
           ('up', 'column'), ('gate', 'column'), ('down', 'row')]


def target_specifications(model, scope='all_expansions'):
    if scope not in ('all_expansions', 'matched_final'):
        raise ValueError(scope)
    records = [r for r in structure(model.config.widths) if r['R']]
    if scope == 'matched_final':
        records = [r for r in records if r['layer'] == len(model.config.widths) - 1]
    targets = []
    for record in records:
        for suffix, axis in TARGETS:
            name = f"blocks.{record['layer']}.{suffix}"
            linear = model.get_submodule(name)
            m, n = linear.weight.shape
            dimension = n if axis == 'column' else m
            assert sorted(record['C'] + record['R']) == list(range(dimension))
            targets.append({'module': name, 'layer': record['layer'], 'axis': axis,
                            'm': m, 'n': n, 'C': record['C'], 'R': record['R'],
                            'residual_width': record['width'], 'R_ages': sorted(set(record['R_ages']))})
    return targets


def grouped(targets, shuffle_seed=None):
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    result = []
    for target in targets:
        size = target['n'] if target['axis'] == 'column' else target['m']
        split = len(target['C'])
        permutation = rng.permutation(size).tolist() if rng is not None else None
        for label, indices in [('C', target['C']), ('R', target['R'])]:
            if permutation is not None:
                indices = sorted(permutation[:split] if label == 'C' else permutation[split:])
            m = len(indices) if target['axis'] == 'row' else target['m']
            n = len(indices) if target['axis'] == 'column' else target['n']
            result.append({'id': target['module'] + ':' + label, 'module': target['module'],
                'layer': target['layer'], 'axis': target['axis'], 'label': label, 'indices': indices,
                'm': m, 'n': n, 'cost': m + n, 'residual_width': target['residual_width']})
    return result


def full_groups(targets):
    return [{'id': t['module'], 'module': t['module'], 'm': t['m'], 'n': t['n'],
             'cost': t['m'] + t['n'], 'residual_width': t['residual_width']} for t in targets]


def integer_rank_balanced(groups, budget, cap, width_proportional=False):
    weights = {g['id']: float(g['residual_width']) if width_proportional else 1. for g in groups}
    unit = budget / sum(g['cost'] * weights[g['id']] for g in groups)
    utilities = {g['id']: [-(r - unit * weights[g['id']]) ** 2 for r in range(cap + 1)] for g in groups}
    return allocate(groups, utilities, budget, cap)


def exact_feasible(groups, budget, cap, force_id=None):
    states = {0}
    for group in groups:
        low = 1 if group['id'] == force_id else 0
        high = min(cap, group['m'], group['n'])
        states = {spent + r * group['cost'] for spent in states for r in range(low, high + 1)
                  if spent + r * group['cost'] <= budget}
    return budget in states


def eligibility_by_cost(groups, budget, cap):
    examples = {g['cost']: g['id'] for g in groups}
    return {str(cost): exact_feasible(groups, budget, cap, gid) for cost, gid in examples.items()}


def common_low_budgets(full, groups, lower, upper, quantum, cap):
    accepted = []
    for budget in range(lower, upper + 1, quantum):
        a, b = eligibility_by_cost(full, budget, cap), eligibility_by_cost(groups, budget, cap)
        if all(a.values()) and all(b.values()):
            accepted.append(budget)
    return accepted


def strict_uniform(groups, budget, cap):
    cost = sum(g['cost'] for g in groups)
    if budget % cost:
        return None
    rank = budget // cost
    if rank < 1 or any(rank > min(cap, g['m'], g['n']) for g in groups):
        return None
    return {g['id']: rank for g in groups}


def module_parameters(groups, ranks, modules):
    values = defaultdict(int)
    for group in groups:
        values[group['module']] += group['cost'] * ranks.get(group['id'], 0)
    return {name: values[name] for name in modules}


def parameter_tv(left, right, budget):
    if sum(left.values()) != budget or sum(right.values()) != budget:
        raise ValueError('Compare exact equal actual budgets only')
    return sum(abs(left.get(k, 0) - right.get(k, 0)) for k in set(left) | set(right)) / (2 * budget)
