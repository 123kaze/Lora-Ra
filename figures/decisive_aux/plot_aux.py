"""Plot observed expanded-scope capacity curves, grouped by independent base."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    bundle = parser.parse_args().bundle.resolve()
    cfg = json.loads((bundle/'config.json').read_text(encoding='utf-8'))
    with (bundle/'summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    output = ROOT/'figures'/bundle.name
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'savefig.dpi': 450, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, profile in zip(axes, cfg['widths']):
        for step, color in [(180, '#0077BB'), (600, '#EE7733')]:
            values = sorted([r for r in rows if r['profile'] == profile and int(r['step']) == step], key=lambda r: int(r['budget']))
            ax.errorbar([int(r['budget']) for r in values], [float(r['val_loss_mean']) for r in values],
                        yerr=[float(r['base_mean_sample_std']) for r in values], color=color, marker='o',
                        lw=1, capsize=2, markersize=3, label=f'{step} steps')
        ax.set_xticks([1216, 2464, 4928], ['1216\nsparse', '2464\nall rank 1', '4928\nall rank 2'])
        ax.set_xlabel('Trainable parameters (seven full matrices)')
        ax.set_title(profile.capitalize()+' bypass')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Validation loss (nat/byte)')
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle('Discovery capacity: 3 old bases; mean ± SD of two-adapter base means', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'capacity.{suffix}', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
