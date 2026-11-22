"""Package the real stage-0 calibration and old-case diagnostic, without tensors."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export(run):
    bundle = ROOT/'results'/run.name
    if bundle.exists():
        raise FileExistsError(bundle)
    bundle.mkdir(parents=True)
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.suffix not in ('.pt', '.pyc') and '__pycache__' not in path.parts:
            destination = bundle/path.relative_to(run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    legacy = read(run/'previous_reversal_diagnostic/input_manifest.json')['files']
    prefix = Path('results/crossdomain_labels_20261122_01')
    for item in legacy:
        source = ROOT/item['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
        destination = bundle/'legacy_inputs'/Path(item['path']).relative_to(prefix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (bundle/'README.md').write_text(
        '# 第0步：离线扫描及2026反向诊断\n\n'
        '6个既有基础，72个FP32校准批次，没有新微调。2160个分配、20160条双侧谱前缀。'
        '原指定预算存在严格基线不可行与成本族不可达；用户随后选择整数预算均衡与末层七类主范围，'
        '1216/2464为相应候选。`resolved_specification.json`记录用户选择和正式协议散列；'
        '`selection.json`保存选择前的历史状态，不应将当时的待决状态误作当前执行状态。\n\n'
        '查看`shapes.csv`、`budget_feasibility.csv`、`nondegeneracy_summary.csv`和`allocations.json`。'
        'V只为预算分布几何参考，不是1216/2464下可实现的VBalanced方法。'
        '`previous_reversal_diagnostic/`为旧结果事后诊断，不证明机制或预测规律。'
        '`legacy_inputs/`包含复核该诊断所需的旧NPZ和CSV，无需torch。\n\n'
        '完整复核：`python scripts/verify_decisive_preflight.py results/' + run.name + '`。'
        '复跑旧案例：`python scripts/analyze_2026_reversal.py --input results/' + run.name + '/legacy_inputs --output <fresh-output>`。\n', encoding='utf-8')
    print(bundle)


def manifest(bundle):
    paths = [p for p in bundle.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json']
    names = ['scripts/export_decisive_preflight.py', 'scripts/verify_decisive_preflight.py',
        'scripts/verify_output_diagonal.py', 'scripts/analyze_2026_reversal.py', 'scripts/summarize_preflight.py',
        'configs/decisive_preflight.json', 'configs/decisive_stage0.json',
        'plan/decisive-stage0-roadmap.md', 'plan/decisive-stage0-protocol-v1.md',
        'tables/decisive-preflight-schema.md', 'reports/decisive_preflight_20261122.md',
        'reports/decisive_stage0_tests.txt', 'reports/decisive_prefix_equivalence.json',
        'figures/decisive_preflight/plot_preflight.py']
    paths += [ROOT/n for n in names if (ROOT/n).exists()]
    paths += [p for p in (ROOT/'figures'/bundle.name).glob('*') if p.is_file()]
    files = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))]
    (bundle/'artifact-manifest.json').write_text(json.dumps({'files': files}, indent=2), encoding='utf-8')
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
