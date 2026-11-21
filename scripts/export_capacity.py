"""Export actual capacity trajectories, logs and source snapshots without tensors."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(run):
    bundle = ROOT / 'results' / run.name
    if bundle.exists():
        raise FileExistsError('Use a new bundle rather than overwriting published evidence')
    bundle.mkdir(parents=True)
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.suffix not in ('.pt', '.pyc') and '__pycache__' not in path.parts:
            destination = bundle / path.relative_to(run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    with (bundle / 'main.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    sets = defaultdict(list)
    for row in rows:
        sets[(row['profile'], int(row['base_seed']), int(row['rank']), int(row['budget']), int(row['step']))].append(row)
    summary = []
    for (profile, seed, rank, budget, step), values in sorted(sets.items()):
        row = {'profile': profile, 'base_seed': seed, 'rank': rank, 'budget': budget,
               'step': step, 'n_adapter_seeds': len(values)}
        for key in ('val_loss', 'train_eval_loss', 'val_improvement'):
            data = [float(v[key]) for v in values]
            row[key + '_mean'] = statistics.mean(data)
            row[key + '_sample_std'] = statistics.stdev(data) if len(data) > 1 else 0.
        summary.append(row)
    with (bundle / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    gate = json.loads((bundle / 'capacity_gate.json').read_text(encoding='utf-8'))
    (bundle / 'README.md').write_text(
        '# 跨域容量筛查真实数据\n\n'
        f'运行：`{run.name}`。容量门结果：`{gate["pass"]}`。本轮没有比较CarryRank方法，没有读取WikiText test。\n\n'
        '`main.csv`包含全部定步记录，`summary.csv`按单个基础、宽度、rank、步数聚合adapter seeds，std为样本标准差。'
        '`capacity_gate.json`保留预定rank1/rank4配对结果；`checkpoint_audit.json`为独立末步重载；'
        '各基础目录保存逐次训练日志和adapter result.json，tensor未包含在此轻量包。\n\n'
        '无torch复核：`python scripts/verify_capacity.py results/' + run.name + '`。'
        '从数据源重建缓存：`python scripts/prepare_wikitext.py`（需要pyarrow）。'
        '训练：`PYTHONPATH=src python -m carryrank.capacity --output runs/<fresh-id>`。\n', encoding='utf-8')
    print(json.dumps({'bundle': str(bundle), 'rows': len(rows), 'summary_rows': len(summary), 'gate': gate['pass']}))
    return bundle


def manifest(bundle):
    # Run after the report and real-data figures have been rendered and reviewed.
    paths = [p for p in bundle.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json']
    for name in ('scripts/prepare_wikitext.py', 'scripts/audit_capacity.py', 'scripts/export_capacity.py',
                 'scripts/verify_capacity.py', 'configs/crossdomain_capacity.json',
                 'plan/crossdomain-capacity-protocol.md', 'tables/crossdomain-capacity-schema.md',
                 'reports/crossdomain_capacity_20261122.md', 'reports/crossdomain_capacity_tests.txt',
                 'figures/crossdomain_capacity/plot_capacity.py'):
        if (ROOT / name).exists():
            paths.append(ROOT / name)
    paths += [p for p in (ROOT / 'figures' / bundle.name).glob('*') if p.is_file()]
    records = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))]
    (bundle / 'artifact-manifest.json').write_text(json.dumps({'files': records}, indent=2), encoding='utf-8')
    print(json.dumps({'files_hashed': len(records)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    if args.manifest_only:
        manifest(args.path.resolve())
    else:
        export(args.path.resolve())


if __name__ == '__main__':
    main()
