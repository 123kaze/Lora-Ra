"""Export a completed real-model run, excluding tensor checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export(run):
    completion = read(run/'completion.json')
    audit = read(run/'checkpoint_audit.json')
    assert completion['complete'] and not completion['smoke']
    assert audit['complete'] and audit['pass'] and audit['main_checkpoints_reloaded'] == 40
    assert (run/'interpretation.json').exists()
    bundle = ROOT/'results'/run.name
    assert not bundle.exists(), 'Use a new bundle or only regenerate its manifest'
    shutil.copytree(run, bundle, ignore=shutil.ignore_patterns('*.pt', '*.pyc', '__pycache__'))
    if (ROOT/'.gitattributes').exists():
        shutil.copy2(ROOT/'.gitattributes', bundle/'transport.gitattributes')
    (bundle/'README.md').write_text(
        '# GPT-2两任务实际实验数据\n\n'
        '只包含完成的非smoke运行。main.csv为40条正式训练，tuning.csv为24条独立调参；'
        '只有一个公开预训练checkpoint，每任务5个adapter/data seed。\n\n'
        '查看[report.md](report.md)、[summary.csv](summary.csv)、'
        '[confidence_intervals.csv](confidence_intervals.csv)、[interpretation.json](interpretation.json)。'
        '逐样本标签logprob、预测与训练日志在units/；训练校准矩阵、谱和分配在calibration/。'
        'selection_lock.json证明正式seed前固定的LR选择；测试集未下载。\n\n'
        'AdaLoRA匹配最终有效因子rank预算，包含额外E标量且峰值物理参数约2倍；'
        '不称全程同物理参数或同计算成本。CoverageFloor若与Spectral同分配会明确标注，'
        '即使独立重训也不增加机制证据。\n\n'
        '全部40个正式adapter已独立GPU重载；.pt和原模型权重只在本地runs/data保留。'
        '本轻量包可无torch从预测重算指标、CI以及从FP32校准矩阵重算谱和rank；'
        '无权重的核验不等于独立重新训练或重载模型。\n\n'
        '`python scripts/verify_realmodel_evidence.py results/'+run.name+'`\n', encoding='utf-8')
    print({'bundle': str(bundle)})


def manifest(bundle):
    files = [p for p in bundle.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json']
    names = ['scripts/realmodel_experiment.py', 'scripts/prepare_realmodel_assets.py',
             'scripts/prepare_realmodel_data.py', 'scripts/analyze_realmodel.py',
             'scripts/verify_realmodel_evidence.py', 'scripts/export_realmodel.py',
             'scripts/report_realmodel.py', 'figures/realmodel/plot_results.py',
             'scripts/profile_realmodel_calibration.py', 'scripts/verify_realmodel_clean_checkout.py',
             'tables/realmodel-schema.md',
             'scripts/run_realmodel.ps1',
             'scripts/finalize_realmodel.ps1',
             'configs/realmodel_gpt2.json', 'plan/realmodel-gpt2-protocol-v1.md',
             'reports/realmodel_environment_20261122.json',
             'reports/realmodel_analysis_tests_20261122.txt',
             'reports/realmodel_forward_tests_20261122.txt',
             'tests/test_realmodel_forward.py',
             'reports/realmodel_gpt2_20261122.md']
    files += [ROOT/name for name in names if (ROOT/name).exists()]
    files += [p for p in (ROOT/'figures'/bundle.name).glob('*') if p.is_file()]
    payload = [{'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(set(files))]
    (bundle/'artifact-manifest.json').write_text(json.dumps({'files': payload}, indent=2), encoding='utf-8')
    print({'files_hashed': len(payload)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    manifest(args.path.resolve()) if args.manifest_only else export(args.path.resolve())
