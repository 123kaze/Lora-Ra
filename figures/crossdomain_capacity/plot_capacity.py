"""Plot measured cross-domain capacity responses, no fitted or mock values."""
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


def load(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    config = json.loads((bundle / 'config.json').read_text(encoding='utf-8'))
    rows = load(bundle / 'summary.csv')
    raw = load(bundle / 'main.csv')
    output = ROOT / 'figures' / bundle.name
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.dpi': 450})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, profile in zip(axes, config['widths']):
        for step, color in zip(config['eval_steps'], COLORS):
            values = sorted([r for r in rows if r['profile'] == profile and int(r['step']) == step], key=lambda r: int(r['rank']))
            ax.errorbar([int(r['budget']) for r in values], [float(r['val_loss_mean']) for r in values],
                        yerr=[float(r['val_loss_sample_std']) for r in values], marker='o', markersize=3,
                        lw=1.2, capsize=2, color=color, label=f'{step} steps')
        age = 5 if profile == 'long' else 1
        ax.set_title(f'{profile.capitalize()} bypass (age {age})')
        ax.set_xscale('log', base=2)
        budgets = [384 * rank for rank in config['ranks']]
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel('Trainable parameters (both V groups)')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('WikiText validation loss (nat/byte)')
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle('Capacity screen: 1 base seed / profile; mean ± adapter sample SD', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'capacity.{suffix}', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, profile in zip(axes, config['widths']):
        for index, seed in enumerate(config['adapter_seeds']):
            values = {int(r['rank']): float(r['val_loss']) for r in raw
                      if r['profile'] == profile and int(r['adapter_seed']) == seed and int(r['step']) == 600}
            ax.plot([1, 4], [values[1], values[4]], marker='o', color=COLORS[index], lw=1,
                    label=f'adapter {seed}')
        ax.set_xticks([1, 4], ['rank 1', 'rank 4'])
        ax.set_title(profile.capitalize() + ' bypass')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('600-step validation loss (nat/byte)')
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle('Locked capacity contrast; lower loss is better', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'paired_gate.{suffix}', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
