"""Export completed stage-0 phase/analysis directories and audit manifests."""
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


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def csv_write(path, values):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def export(run):
    if (run/'worker_links.json').exists():
        kind = 'main_analysis'
        assert (run/'verdict.json').exists()
    else:
        complete = read(run/'completion.json')
        assert complete['complete'] and not complete['smoke']
        kind = complete['phase']
        assert read(run/'checkpoint_audit.json')['complete_run']
    bundle = ROOT/'results'/run.name
    if bundle.exists():
        raise FileExistsError(bundle)
    shutil.copytree(run, bundle, ignore=shutil.ignore_patterns('*.pt', '*.pyc', '__pycache__'))
    if kind != 'main_analysis':
        grouped = defaultdict(list)
        for row in rows(bundle/'main.csv'):
            grouped[(row['profile'], int(row['base_seed']), int(row['budget']), row['method'], int(row['step']))].append(float(row['val_loss']))
        base = [{'profile': p, 'base_seed': s, 'budget': b, 'method': m, 'step': t,
                 'adapter_count': len(v), 'val_loss_mean': statistics.mean(v)}
                for (p, s, b, m, t), v in sorted(grouped.items())]
        csv_write(bundle/'by_base_summary.csv', base)
        grand = defaultdict(list)
        for row in base:
            grand[(row['profile'], row['budget'], row['method'], row['step'])].append(row['val_loss_mean'])
        csv_write(bundle/'summary.csv', [{'profile': p, 'budget': b, 'method': m, 'step': t,
            'base_count': len(v), 'val_loss_mean': statistics.mean(v),
            'base_mean_sample_std': statistics.stdev(v) if len(v) > 1 else 0.}
            for (p, b, m, t), v in sorted(grand.items())])
        if kind == 'capacity':
            shutil.copy2(bundle/'main.csv', bundle/'sweep.csv')
    (bundle/'README.md').write_text(
        '# 第0步真实证据包\n\n'
        f'运行`{run.name}`，类型`{kind}`。主表和逐单位日志来自完整运行；所有.pt只保留本地。'
        'protocol.md及源快照记录训练前约定，不以代码、计划或smoke作为实验结果。\n\n'
        'FullBudgetBalanced是整数预算均衡的整矩阵LoRA；2464恰为末层7矩阵全rank1。'
        'GroupedBudgetBalanced不等同于无法在这些预算上实现的严格全组统一rank。'
        'WidthRankProportional是已证明等价的别名，复用来源在conditions.csv/reused_from中明确记录。\n\n'
        '标准差先平均adapter，再跨基础均值。主试验10个新基础；容量与单组探针仅3个发现基础，分开报告。'
        '基础config.json中的pilot adapter/sweep字段未调用，实际适配使用运行根config.json、协议和单位分配。'
        '联合CI规则不按中途结果改动，阴性不是等效性证明。\n\n'
        '主分析可用`python scripts/verify_decisive_main.py results/' + run.name + '`无torch复核。'
        '容量/机制阶段可用`python scripts/verify_decisive_phase.py results/' + run.name + '`。'
        '有完整报告时请查看`report.md`。\n', encoding='utf-8')
    print({'bundle': str(bundle), 'kind': kind})


def manifest(bundle):
    paths = [p for p in bundle.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json']
    kind = 'main_analysis' if (bundle/'worker_links.json').exists() else read(bundle/'completion.json')['phase']
    names = ['scripts/export_decisive.py', 'scripts/audit_decisive.py',
        'scripts/verify_decisive_phase.py', 'configs/decisive_stage0.json',
        'configs/decisive_stage0_long.json', 'configs/decisive_stage0_short.json',
        'plan/decisive-stage0-protocol-v1.md', 'tables/decisive-main-schema.md',
        'reports/decisive_stage0_tests.txt', 'reports/decisive_analysis_tests.txt',
        'reports/decisive_prefix_equivalence.json']
    if kind == 'main_analysis':
        names += ['scripts/verify_decisive_main.py', 'scripts/verify_output_diagonal.py',
                  'scripts/analyze_decisive_main.py', 'scripts/report_decisive.py', 'figures/decisive_main/plot_decisive.py']
    elif kind == 'mechanism':
        names += ['scripts/analyze_decisive_mechanism.py', 'scripts/verify_decisive_mechanism.py']
    else:
        names += ['figures/decisive_aux/plot_aux.py']
    paths += [ROOT/n for n in names if (ROOT/n).exists()]
    paths += [p for p in (ROOT/'figures'/bundle.name).glob('*') if p.is_file()]
    # Reports are immutable dated outputs, not the live execution-state file.
    names = {'decisive_capacity_20261122_01': 'reports/decisive_capacity_20261122.md',
             'decisive_main_20261122_01': 'reports/decisive_stage0_20261122.md',
             'decisive_mechanism_20261122_01': 'reports/decisive_mechanism_20261122.md'}
    if bundle.name in names and (ROOT/names[bundle.name]).exists():
        paths.append(ROOT/names[bundle.name])
    files = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))]
    write(bundle/'artifact-manifest.json', {'files': files})
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
