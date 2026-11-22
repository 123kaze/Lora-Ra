"""Plot completed real GPT-2 evidence; no model execution or invented values."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('UniformLoRA', 'AdaLoRA', 'TwoSidedSpectral', 'CoverageFloor')
LABELS = ('Uniform', 'AdaLoRA', 'Spectral', 'Floor')
COLORS = ('#0077BB', '#EE7733', '#009988', '#CC3311')


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def save(fig, out, name):
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'{name}.{suffix}', bbox_inches='tight')
    plt.close(fig)


def main(bundle):
    cfg = json.loads((bundle/'config.json').read_text(encoding='utf-8'))
    interpretation = json.loads((bundle/'interpretation.json').read_text(encoding='utf-8'))
    assert interpretation['complete'] and interpretation['main_units'] == 40
    summary, actual = rows(bundle/'summary.csv'), rows(bundle/'main.csv')
    assert len(actual) == 40 and all(row['phase'] == 'main' for row in actual)
    assert len(summary) == 8
    degenerate = interpretation['coverage_floor_rank_degenerate_tasks']
    identical_note = ('\n'+', '.join(task.upper() for task in degenerate)+': Spectral / Floor ranks identical; separate retraining') if degenerate else ''
    out = ROOT/'figures'/bundle.name
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 450, 'svg.fonttype': 'none', 'legend.frameon': False})
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.1), squeeze=False)
    for row_index, task in enumerate(cfg['tasks']):
        for column, metric in enumerate(('nll', 'accuracy', 'f1')):
            ax = axes[row_index, column]
            for index, (method, color) in enumerate(zip(METHODS, COLORS)):
                cell = next(row for row in summary if (row['task'], row['method']) == (task, method))
                observations = sorted([row for row in actual if (row['task'], row['method']) == (task, method)], key=lambda r: int(r['seed']))
                y = [float(row[metric]) for row in observations]
                x = np.linspace(index-.09, index+.09, len(y))
                ax.scatter(x, y, color=color, s=9, alpha=.45, linewidths=0)
                ax.errorbar(index, float(cell[metric+'_mean']), yerr=float(cell[metric+'_sample_std']),
                            color=color, fmt='o', capsize=2, ms=4, lw=.9)
            ax.set_xticks(range(4), LABELS, rotation=25, ha='right')
            ax.set_title(task.upper())
            ax.set_ylabel({'nll': 'Label NLL (nat)', 'accuracy': 'Accuracy', 'f1': 'Positive-label F1'}[metric])
            ax.grid(axis='y', alpha=.15)
            if metric != 'nll':
                ax.set_ylim(0, 1.02)
    fig.suptitle('One pretrained GPT-2 checkpoint; five paired adapter/data seeds\nDots: actual runs; error bars: mean ± sample SD'+identical_note, fontsize=9)
    save(fig, out, 'comparison')
    intervals = rows(bundle/'confidence_intervals.csv')
    names = ('uniform_minus_spectral', 'spectral_minus_floor', 'uniform_minus_adalora')
    labels = ('Uniform − Spectral [primary]', 'Spectral − Floor', 'Uniform − AdaLoRA')
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), squeeze=False, sharex=True)
    for ax, task in zip(axes[0], cfg['tasks']):
        for index, name in enumerate(names):
            row = next(r for r in intervals if r['task'] == task and r['contrast'] == name)
            mean, low, high = [float(row[key]) for key in ('mean', 'low', 'high')]
            ax.errorbar(mean, index, xerr=np.array([[mean-low], [high-mean]]), fmt='o',
                        color=COLORS[index], capsize=3, ms=4, lw=1)
        ax.axvline(0, color='.45', lw=.8)
        task_labels = list(labels)
        if task in degenerate:
            task_labels[1] += ' [same ranks]'
        ax.set_yticks(range(3), task_labels)
        ax.invert_yaxis(); ax.set_title(task.upper())
        ax.set_xlabel('Paired NLL difference (first − second)')
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Marginal 95% paired-seed bootstrap CI; 200,000 shared-index draws\nPrimary: negative favors Uniform; secondary: positive favors Floor / AdaLoRA', fontsize=9)
    save(fig, out, 'paired_nll')
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), squeeze=False)
    for row_index, task in enumerate(cfg['tasks']):
        for column, metric in enumerate(('rank_hhi', 'nonzero_modules')):
            ax = axes[row_index, column]
            for index, (method, color) in enumerate(zip(METHODS, COLORS)):
                row = next(r for r in summary if (r['task'], r['method']) == (task, method))
                observations = [r for r in actual if (r['task'], r['method']) == (task, method)]
                ax.scatter(np.linspace(index-.08, index+.08, len(observations)),
                           [float(r[metric]) for r in observations], s=9, color=color, alpha=.4)
                ax.errorbar(index, float(row[metric+'_mean']), yerr=float(row[metric+'_sample_std']),
                            color=color, fmt='o', lw=.9, capsize=2, ms=4)
            ax.set_xticks(range(4), LABELS, rotation=20, ha='right')
            ax.set_title(task.upper()); ax.grid(axis='y', alpha=.15)
            if metric == 'rank_hhi':
                ax.set_ylabel('Final rank-share HHI')
                ax.axhline(1/12, ls='--', color='.6', lw=.7)
            else:
                ax.set_ylabel('Final nonzero QKV matrices / 12')
                ax.set_ylim(-.2, 12.8); ax.set_yticks([0, 3, 6, 9, 12])
    fig.suptitle('Observed final allocations; equal 48-rank factor budget\nAdaLoRA has 48 extra effective E scalars and higher physical training cost', fontsize=9)
    save(fig, out, 'allocation')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    main(parser.parse_args().bundle.resolve())
