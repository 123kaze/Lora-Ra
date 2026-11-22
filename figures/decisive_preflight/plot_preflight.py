"""Observed allocation distances and the already known reversal's spectral decay."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = ['#0077BB', '#EE7733', '#009988']


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    bundle = parser.parse_args().bundle.resolve()
    output = ROOT/'figures'/bundle.name
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'savefig.dpi': 450, 'axes.spines.top': False, 'axes.spines.right': False})
    summary = rows(bundle/'nondegeneracy_summary.csv')
    fields = [('TV_vs_full_integer_balanced_minimum', 'Full budget balanced'),
              ('TV_vs_grouped_integer_balanced_minimum', 'Grouped budget balanced'),
              ('TV_vs_V_budget_reference_minimum', 'V budget reference')]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, scope, name in zip(axes, ['matched_final', 'all_expansions'], ['Matched final seven', 'All expansion layers']):
        values = sorted([r for r in summary if r['scope'] == scope], key=lambda r: int(r['budget']))
        for color, (field, label) in zip(COLORS, fields):
            ax.plot([int(r['budget']) for r in values],
                    [float(r[field]) if r[field] else np.nan for r in values],
                    color=color, marker='.', markersize=3, lw=.8, label=label)
        ax.axhline(.30, color='#666666', ls='--', lw=.8)
        for b in [1216, 2464]:
            ax.axvline(b, color='#999999', ls=':', lw=.8)
        ax.set_title(name)
        ax.set_xlabel('Actual parameter budget')
        ax.set_ylim(-.03, 1.03)
        ax.set_xticks([384, 1216, 2464, 3072])
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Minimum module parameter TV\nover 6 discovery bases')
    axes[1].legend(frameon=False, fontsize=6)
    fig.suptitle('Calibration-only screening; distances do not measure validation gains', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'budget_screen.{suffix}', bbox_inches='tight')
    plt.close(fig)
    spectra = rows(bundle/'previous_reversal_diagnostic/full_spectra.csv')
    moments = rows(bundle/'previous_reversal_diagnostic/case_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for seed, color in zip([2026, 2027, 2028], COLORS):
        values = [r for r in spectra if r['profile'] == 'long' and int(r['base_seed']) == seed
                  and r['module'].endswith('v_proj') and r['metric'] == 'diag_both'
                  and r['label'] == 'R' and int(r['rank']) <= 8]
        axes[0].plot([int(r['rank']) for r in values], [float(r['cumulative_energy_fraction']) for r in values],
                     color=color, marker='o', markersize=3, label=f'base {seed}')
    axes[0].set_xlabel('Rank in the old final V/R block')
    axes[0].set_ylabel('Cumulative two-sided spectral energy')
    axes[0].set_title('R spectra are similar, not identical')
    axes[0].legend(frameon=False, fontsize=7)
    long = [r for r in moments if r['profile'] == 'long']
    axes[1].scatter([float(r['V_C_over_R_input_second_moment']) for r in long],
                    [float(r['shuffled_minus_true']) for r in long], color=COLORS, s=25)
    for row in long:
        axes[1].annotate(row['base_seed'], (float(row['V_C_over_R_input_second_moment']),
                         float(row['shuffled_minus_true'])), xytext=(4, 3), textcoords='offset points', fontsize=7)
    axes[1].axhline(0, color='#666666', lw=.7)
    axes[1].set_xlabel('Old final V input E[x²] C/R ratio')
    axes[1].set_ylabel('Shuffled mean minus true loss')
    axes[1].set_title('Three post-hoc cases; no fitted rule')
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'old_reversal.{suffix}', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
