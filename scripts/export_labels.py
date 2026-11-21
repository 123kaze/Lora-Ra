"""Aggregate only completed label-screen measurements; preserve base-level pairing."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows, cfg):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['profile'], int(row['base_seed']), row['method'], int(row['step']))].append(float(row['val_loss']))
    by_base, summary = [], []
    for (profile, seed, method, step), values in sorted(grouped.items()):
        assert len(values) == len(cfg['adapter_seeds'])
        by_base.append({'profile': profile, 'base_seed': seed, 'method': method, 'step': step,
                        'adapter_count': len(values), 'val_loss_mean': statistics.mean(values),
                        'val_loss_adapter_sample_std': statistics.stdev(values) if len(values) > 1 else 0.})
    means = {(r['profile'], r['base_seed'], r['method'], r['step']): r['val_loss_mean'] for r in by_base}
    methods = ['true'] + [f'shuffle_{s}' for s in cfg['shuffle_seeds']] + ['VBalanced']
    comparisons = []
    for profile in cfg['widths']:
        for step in cfg['eval_steps']:
            for method in methods + ['ShuffledMean']:
                values = []
                for seed in cfg['base_seeds']:
                    if method == 'ShuffledMean':
                        value = statistics.mean(means[(profile, seed, f'shuffle_{s}', step)] for s in cfg['shuffle_seeds'])
                    else:
                        value = means[(profile, seed, method, step)]
                    values.append(value)
                summary.append({'profile': profile, 'method': method, 'step': step, 'base_count': len(values),
                                'val_loss_mean': statistics.mean(values),
                                'val_loss_base_sample_std': statistics.stdev(values) if len(values) > 1 else 0.})
            for seed in cfg['base_seeds']:
                actual = means[(profile, seed, 'true', step)]
                shuffled = statistics.mean(means[(profile, seed, f'shuffle_{s}', step)] for s in cfg['shuffle_seeds'])
                fixed = means[(profile, seed, 'VBalanced', step)]
                comparisons.append({'profile': profile, 'base_seed': seed, 'step': step, 'true_loss': actual,
                                    'shuffled_mean_loss': shuffled, 'VBalanced_loss': fixed,
                                    'shuffled_minus_true': shuffled - actual, 'VBalanced_minus_true': fixed - actual})
    verdict = {'smoke': cfg['smoke'], 'primary_step': cfg['primary_step'], 'threshold': cfg['minimum_label_advantage'],
               'primary_profile': 'long', 'test_split_used': False, 'by_profile': {}}
    for profile in cfg['widths']:
        values = [r for r in comparisons if r['profile'] == profile and r['step'] == cfg['primary_step']]
        checks = {}
        for contrast in ('shuffled_minus_true', 'VBalanced_minus_true'):
            delta = [r[contrast] for r in values]
            checks[contrast] = {'paired_base_differences': delta, 'mean': statistics.mean(delta),
                'sample_std': statistics.stdev(delta) if len(delta) > 1 else 0.,
                'pass': not cfg['smoke'] and all(x > 0 for x in delta) and statistics.mean(delta) >= cfg['minimum_label_advantage']}
        verdict['by_profile'][profile] = checks
    verdict['primary_label_gate_pass'] = verdict['by_profile']['long']['shuffled_minus_true']['pass']
    verdict['primary_method_gate_pass'] = verdict['by_profile']['long']['VBalanced_minus_true']['pass']
    return by_base, summary, comparisons, verdict


def export(run):
    if not read(run / 'completion.json')['complete']:
        raise ValueError('Incomplete run')
    cfg = read(run / 'config.json')
    if cfg['smoke']:
        raise ValueError('This exporter is for the full prespecified method set')
    bundle = ROOT / 'results' / run.name
    if bundle.exists():
        raise FileExistsError(bundle)
    bundle.mkdir(parents=True)
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.suffix not in ('.pt', '.pyc') and '__pycache__' not in path.parts:
            destination = bundle / path.relative_to(run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        destination = bundle / 'source_snapshot/tests' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    by_base, summary, comparisons, verdict = analyze(csv_rows(bundle / 'main.csv'), cfg)
    csv_write(bundle / 'by_base_summary.csv', by_base)
    csv_write(bundle / 'summary.csv', summary)
    csv_write(bundle / 'base_comparisons.csv', comparisons)
    write(bundle / 'verdict.json', verdict)
    assignments = []
    for profile in cfg['widths']:
        for seed in cfg['base_seeds']:
            child = bundle / f'{profile}_base_{seed}'
            schemes = read(child / 'groups.json')
            for method, item in read(child / 'allocations.json').items():
                groups = schemes[item['scheme']]
                ranks = item['ranks']
                assignments.append({'profile': profile, 'base_seed': seed, 'method': method,
                    'Q_parameters': sum(g['cost'] * ranks[g['id']] for g in groups if g['module'].endswith('q_proj')),
                    'V_parameters': sum(g['cost'] * ranks[g['id']] for g in groups if g['module'].endswith('v_proj')),
                    'first_half_rank': sum(ranks[g['id']] for g in groups if g['label'] == 'C'),
                    'second_half_rank': sum(ranks[g['id']] for g in groups if g['label'] == 'R'),
                    'same_ranks_and_groups_as_VBalanced': item['scheme'] == 'true' and ranks == read(child / 'allocations.json')['VBalanced']['ranks']})
    csv_write(bundle / 'allocation_summary.csv', assignments)
    (bundle / 'README.md').write_text(
        '# 跨域真实C/R与五个随机分组\n\n'
        '84条600步轨迹、168条180/600步验证记录。长/短旁路、3个基础seed、2个adapter seed、预算384、7种方法。'
        '本轮没有读取test，不能视作确认性测试。\n\n'
        '`main.csv`是全部原始记录；`by_base_summary.csv`先平均adapter；`summary.csv`中的标准差只跨3个基础均值；'
        '`ShuffledMean`先对每个基础的5个随机分组取平均。`base_comparisons.csv`和`verdict.json`保留全部基础配对差。'
        '不要把随机分组seed乘adapter seed当独立基础样本。\n\n'
        '`allocation_summary.csv`和各基础`groups.json/allocations.json`保留分配与列索引。'
        '随机组中的C/R键只是槽位名称；真实语义仅属于true组。`calibration_arrays.npz`不含权重，NumPy可读取；'
        '`checkpoint_audit.json`给出独立重载；逐步JSONL和源码快照均保留。\n\n'
        '无torch完整统计/SVD复核：`python scripts/verify_labels.py results/' + run.name + '`。\n', encoding='utf-8')
    print({'bundle': str(bundle), 'rows': len(csv_rows(bundle / 'main.csv')), 'verdict': verdict})


def manifest(bundle):
    paths = [p for p in bundle.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json']
    names = ['scripts/export_labels.py', 'scripts/report_labels.py', 'scripts/audit_labels.py', 'scripts/verify_labels.py',
             'scripts/verify_output_diagonal.py', 'scripts/verify_capacity.py', 'scripts/prepare_wikitext.py',
             'configs/crossdomain_labels.json', 'plan/crossdomain-label-protocol.md',
             'tables/crossdomain-label-schema.md', 'reports/crossdomain_labels_20261122.md',
             'reports/crossdomain_labels_tests.txt', 'figures/crossdomain_labels/plot_labels.py']
    paths += [ROOT / name for name in names if (ROOT / name).exists()]
    paths += list((ROOT / 'figures' / bundle.name).glob('*'))
    files = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths)) if p.is_file()]
    write(bundle / 'artifact-manifest.json', {'files': files})
    print({'files_hashed': len(files)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    if args.manifest_only:
        manifest(args.path.resolve())
    else:
        export(args.path.resolve())
