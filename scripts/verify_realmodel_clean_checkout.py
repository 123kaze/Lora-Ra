"""Check the exported real-model evidence in a temporary NumPy-only checkout."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main(bundle):
    assert importlib.util.find_spec('torch') is None, 'Use the CPU NumPy environment without torch'
    relative_bundle = bundle.relative_to(ROOT)
    transport = {'git_repository_present': (ROOT/'.git').exists(), 'checked': False}
    if transport['git_repository_present']:
        samples = [relative_bundle/'main.csv', relative_bundle/'bootstrap.json', relative_bundle/'report.md']
        for sample in samples:
            raw = subprocess.check_output(['git', 'hash-object', '--no-filters', str(sample)], cwd=ROOT, text=True).strip()
            filtered = subprocess.check_output(['git', 'hash-object', '--path='+sample.as_posix(), str(sample)], cwd=ROOT, text=True).strip()
            assert raw == filtered, ('Git would change evidence bytes', sample)
            assert subprocess.run(['git', 'check-ignore', '-q', str(sample)], cwd=ROOT).returncode == 1
        # Pattern check only: the public evidence checkout need not contain runs.
        raw_tensor = Path('runs')/bundle.name/'units'/'representative'/'adapter.pt'
        assert subprocess.run(['git', 'check-ignore', '-q', str(raw_tensor)], cwd=ROOT).returncode == 0
        transport.update(checked=True, exact_byte_samples=len(samples), review_files_not_ignored=True,
                         tensor_checkpoint_ignored=True)
    parent = Path(tempfile.gettempdir()).resolve()
    scratch = Path(tempfile.mkdtemp(prefix='lorara_realmodel_evidence_', dir=parent)).resolve()
    try:
        assert scratch.parent == parent and scratch.name.startswith('lorara_realmodel_evidence_')
        manifest_path = bundle/'artifact-manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        names = {Path(row['path']) for row in manifest['files']}|{manifest_path.relative_to(ROOT)}
        for name in sorted(names):
            source, target = (ROOT/name).resolve(), (scratch/name).resolve()
            assert source.is_relative_to(ROOT) and target.is_relative_to(scratch)
            assert source.suffix != '.pt'
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        assert not (scratch/'runs').exists()
        environment = os.environ.copy()
        environment.update({'PYTHONPATH': '', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        command = [sys.executable, str(scratch/'scripts/verify_realmodel_evidence.py'), str(scratch/relative_bundle)]
        checked = subprocess.run(command, cwd=scratch, env=environment, capture_output=True, text=True, timeout=300)
        assert checked.returncode == 0, checked.stderr
        print('Clean NumPy-only evidence verification passed', flush=True)
        target_bundle = scratch/relative_bundle
        summary = target_bundle/'summary.csv'
        with summary.open(newline='', encoding='utf-8') as handle:
            values = list(csv.DictReader(handle))
        values[0]['nll_mean'] = str(float(values[0]['nll_mean'])+.123)
        with summary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        updated = json.loads((target_bundle/'artifact-manifest.json').read_text())
        item = next(row for row in updated['files'] if row['path'] == summary.relative_to(scratch).as_posix())
        item.update(bytes=summary.stat().st_size, sha256=hashlib.sha256(summary.read_bytes()).hexdigest())
        (target_bundle/'artifact-manifest.json').write_text(json.dumps(updated, indent=2), encoding='utf-8')
        rejected = subprocess.run(command, cwd=scratch, env=environment, capture_output=True, text=True, timeout=300)
        assert rejected.returncode != 0 and 'AssertionError' in rejected.stderr
        output = {'pass': True, 'requires_torch': False, 'runs_directory_present': False,
            'git_transport_and_visibility': transport,
            'files_copied': len(names), 'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'clean_stdout': checked.stdout, 'clean_stderr': checked.stderr,
            'modified_summary_with_refreshed_checksum_rejected': True,
            'rejection_stderr': rejected.stderr[-1800:]}
        destination = ROOT/'reports/realmodel_clean_checkout_20261122.json'
        destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
        print({'pass': True, 'tamper_rejected': True}, flush=True)
    finally:
        resolved = scratch.resolve()
        assert resolved.parent == parent and resolved.name.startswith('lorara_realmodel_evidence_')
        shutil.rmtree(resolved)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    main(parser.parse_args().bundle.resolve())
