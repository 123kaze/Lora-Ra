"""Compatibility entry for the interrupted phase-2 prototype.

The recovered implementation is preserved in
runs/diagnostic_20261122_01/recovered_diagnostics_prototype.py.
Active implementation: carryrank.diagnostic.
"""
from .diagnostic import input_utility, main, rank_scale


def scaling_for(rank, mode):
    return rank_scale(mode, rank)


if __name__ == "__main__":
    main()
