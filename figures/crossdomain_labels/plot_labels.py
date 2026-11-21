"""Measured label results: base-level dispersion and predeclared paired contrasts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = ['#0077BB', '#EE7733', '#009988', '#CC3311']


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    cfg = json.loads((bundle / 'config.json').read_text(encoding='utf-8'))
    summary = rows(bundle / 'summary.csv')
    comparisons = rows(bundle / 'base_comparisons.csv')
    output = ROOT / 'figures' / bundle.name
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none',
                         'savefig.dpi': 450, 'axes.spines.top': False, 'axes.spines.right': False})
    methods = ['true', 'ShuffledMean', 'VBalanced']
    names = ['True C/R', '5 shuffled\npartitions', 'Fixed V\nbalanced']
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    ranges = []
    for ax, profile in zip(axes, cfg['widths']):
        values = {r['method']: r for r in summary if r['profile'] == profile and int(r['step']) == 600}
        for index, method in enumerate(methods):
            ax.errorbar(index, float(values[method]['val_loss_mean']),
                        yerr=float(values[method]['val_loss_base_sample_std']),
                        fmt='o', color=COLORS[index], capsize=3, markersize=5)
        ax.set_xticks(range(3), names)
        ax.set_title(profile.capitalize() + ' bypass')
        low = min(float(values[m]['val_loss_mean']) - float(values[m]['val_loss_base_sample_std']) for m in methods)
        high = max(float(values[m]['val_loss_mean']) + float(values[m]['val_loss_base_sample_std']) for m in methods)
        ranges.append((low - .03, high + .03))
        ax.set_xlim(-.4, 2.4)
        for index, method in enumerate(methods):
            ax.text(index, float(values[method]['val_loss_mean']) + float(values[method]['val_loss_base_sample_std']) + .007,
                    f"{float(values[method]['val_loss_mean']):.4f}", ha='center', fontsize=7)
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('600-step validation loss (nat/byte)')
    lower = min(value[0] for value in ranges)
    upper = max(value[1] for value in ranges)
    axes[0].set_ylim(lower, upper)
    fig.suptitle('Budget 384; mean ± SD across 3 base means (2 adapters each)', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'comparison.{suffix}', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, profile in zip(axes, cfg['widths']):
        selected = [r for r in comparisons if r['profile'] == profile and int(r['step']) == 600]
        for index, row in enumerate(selected):
            jitter = (index - 1) * .06
            ax.scatter([jitter, 1 + jitter], [float(row['shuffled_minus_true']), float(row['VBalanced_minus_true'])],
                       color=COLORS[index], s=25, label=f"base {row['base_seed']}", zorder=3)
        ax.axhline(0, color='#666666', lw=.7)
        ax.axhline(.01, color='#666666', lw=.9, linestyle='--', label='0.01 threshold')
        ax.set_xlim(-.4, 1.4)
        ax.set_xticks([0, 1], ['Shuffled mean\nminus true', 'V balanced\nminus true'])
        ax.set_title(profile.capitalize() + ' bypass')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Paired base difference (nat/byte)\npositive favors true C/R')
    axes[1].legend(frameon=False, fontsize=6.5, loc='best')
    fig.suptitle('Predeclared label and fixed-baseline contrasts at 600 steps', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'paired_differences.{suffix}', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
