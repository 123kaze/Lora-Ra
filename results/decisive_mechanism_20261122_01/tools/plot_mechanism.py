"""Plot real single-group rank predictions and cost-stratified correlations."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main(bundle):
    data = rows(bundle/'single_group_results.csv')
    corr = rows(bundle/'spearman_by_base.csv')
    assert len(data) == 84
    out = ROOT/'figures'/bundle.name
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 450, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True, sharey=True)
    for row_idx, profile in enumerate(('long', 'short')):
        for col_idx, seed in enumerate((2026, 2027, 2028)):
            ax = axes[row_idx, col_idx]
            for cost, marker, color in ((192, 'o', '#0077BB'), (416, '^', '#EE7733')):
                selected = [r for r in data if r['profile'] == profile and int(r['base_seed']) == seed and int(r['budget']) == cost]
                ax.scatter([float(r['proxy_rank_within_cost']) for r in selected],
                           [float(r['actual_rank_within_cost']) for r in selected],
                           marker=marker, color=color, s=20, label=f'{cost} parameters')
            ax.plot([1, 8], [1, 8], color='.65', lw=.7, ls='--')
            ax.set_title(f'{profile}, base {seed}')
            ax.set_xticks([1, 4, 8]); ax.set_yticks([1, 4, 8])
            if row_idx == 1:
                ax.set_xlabel('Predicted rank (higher is better)')
            if col_idx == 0:
                ax.set_ylabel('Measured improvement rank')
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.suptitle('84 separate single-group runs; rankings within equal parameter cost', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out/f'prediction.{ext}', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, profile in zip(axes, ('long', 'short')):
        for cost, color, offset in ((192, '#0077BB', -.05), (416, '#EE7733', .05)):
            selected = [r for r in corr if r['profile'] == profile and r['cost_stratum'] == str(cost)]
            ax.plot([int(r['base_seed'])+offset for r in selected], [float(r['spearman']) for r in selected],
                    'o-', color=color, lw=.8, markersize=4, label=f'{cost} parameters')
        ax.axhline(0, color='.6', lw=.7)
        ax.set_xticks([2026, 2027, 2028]); ax.set_ylim(-.1, 1.05)
        ax.set_title(profile.capitalize()+' bypass'); ax.set_xlabel('Discovery base seed')
    axes[0].set_ylabel('Spearman correlation')
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle('Descriptive only: 3 old bases, adapter 11; excluded from A/B gate', fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out/f'sensitivity.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    main(parser.parse_args().bundle.resolve())
