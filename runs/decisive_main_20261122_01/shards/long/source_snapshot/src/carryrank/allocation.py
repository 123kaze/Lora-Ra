"""Exact multiple-choice knapsack over rank prefixes; no greedy-optimality claim."""
from __future__ import annotations


def allocate(groups, utilities, budget, max_rank, exact=True):
    # Small pilot: integer parameter counts are inexpensive, no budget quantization.
    states = {0: (0.0, {})}
    for group in groups:
        next_states = {}
        gid, cost = group["id"], group["cost"]
        cap = min(max_rank, group["m"], group["n"])
        for spent, (value, ranks) in states.items():
            for rank in range(cap + 1):
                new_spent = spent + rank * cost
                if new_spent > budget:
                    break
                score = value + utilities[gid][rank]
                if new_spent not in next_states or score > next_states[new_spent][0]:
                    next_states[new_spent] = (score, {**ranks, gid: rank})
        states = next_states
    if exact:
        if budget not in states:
            raise ValueError(f"Budget {budget} is not exactly achievable")
        spent = budget
    else:
        spent = max(states, key=lambda s: (states[s][0], -s))
    value, ranks = states[spent]
    return ranks, spent, value


def equal_parameters(groups, budget, max_rank):
    """Exact budget, minimize squared parameter deviation from equal per-group cost."""
    target = budget / len(groups)
    utilities = {g["id"]: [-((k * g["cost"] - target) / max(target, 1)) ** 2
                            for k in range(max_rank + 1)] for g in groups}
    return allocate(groups, utilities, budget, max_rank)
