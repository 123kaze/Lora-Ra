"""Plot the completed stage-0 paired intervals; no mock or inferred results."""
import argparse
import csv
import json
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
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    cfg = json.loads((bundle/'config.json').read_text(encoding='utf-8'))
    intervals = {r['contrast']: r for r in rows(bundle/'confidence_intervals.csv')}
    summary = rows(bundle/'summary.csv')
    output = ROOT/'figures'/bundle.name
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'savefig.dpi': 450, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6), sharex=True)
    labels = [('long_full', 'Long: full balanced minus true'),
              ('long_shuffle', 'Long: shuffled mean minus true'),
              ('age_full', 'Long minus short: full contrast'),
              ('age_shuffle', 'Long minus short: shuffled contrast')]
    for ax, (key, title) in zip(axes.flat, labels):
        for i, budget in enumerate(cfg['budgets']):
            row = intervals[f'B{budget}_{key}']
            mean, low, high = [float(row[k]) for k in ('mean', 'low', 'high')]
            ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='o', color=COLORS[i], capsize=3, markersize=4)
        ax.axvline(0, color='#666666', lw=.8)
        ax.set_yticks(range(len(cfg['budgets'])), [str(b) for b in cfg['budgets']])
        ax.set_ylim(-.5, len(cfg['budgets'])-.5)
        ax.set_title(title, fontsize=8)
        ax.grid(axis='x', alpha=.15)
    for ax in axes[:, 0]:
        ax.set_ylabel('Parameter budget')
    for ax in axes[-1]:
        ax.set_xlabel('Paired loss difference (nat/byte)')
    fig.suptitle('Stage-0 joint gate: all eight lower 95% bounds must be positive', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'joint_gate.{suffix}', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    methods = ['FullBudgetBalanced', 'GroupedBudgetBalanced', 'TwoSidedTrue', 'ShuffledMean']
    colors = ['#0077BB', '#EE7733', '#009988', '#CC3311']
    for ax, profile in zip(axes, cfg['widths']):
        for method, color in zip(methods, colors):
            values = sorted([r for r in summary if r['profile'] == profile and r['method'] == method and int(r['step']) == 600], key=lambda r: int(r['budget']))
            ax.errorbar([int(r['budget']) for r in values], [float(r['val_loss_mean']) for r in values],
                        yerr=[float(r['base_mean_sample_std']) for r in values], fmt='o-', color=color,
                        lw=1, capsize=2, markersize=3, label=method)
        ax.set_xticks(cfg['budgets'])
        ax.set_xlabel('Actual parameter budget')
        ax.set_title(profile.capitalize()+' bypass')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('600-step validation loss (nat/byte)')
    axes[1].legend(frameon=False, fontsize=6)
    fig.suptitle('Mean ± sample SD across 10 base means; width alias omitted', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'comparison.{suffix}', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
