"""Verify all new evidence without runs, torch or the working source tree."""
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
PACKAGES = {
    'decisive_preflight_20261122_01': 'verify_decisive_preflight.py',
    'decisive_capacity_20261122_01': 'verify_decisive_phase.py',
    'decisive_main_20261122_01': 'verify_decisive_main.py',
    'decisive_mechanism_20261122_01': 'verify_decisive_mechanism.py',
}


def main():
    assert importlib.util.find_spec('torch') is None, 'Run with the existing CPU NumPy environment without torch'
    temp_parent = Path(tempfile.gettempdir()).resolve()
    checkout = Path(tempfile.mkdtemp(prefix='lorara_decisive_evidence_', dir=temp_parent)).resolve()
    assert checkout.parent == temp_parent and checkout.name.startswith('lorara_decisive_evidence_')
    records, required = [], set()
    try:
        for package in PACKAGES:
            relative = Path('results')/package/'artifact-manifest.json'
            manifest = json.loads((ROOT/relative).read_text(encoding='utf-8'))
            required.add(relative)
            required.update(Path(row['path']) for row in manifest['files'])
        for relative in sorted(required):
            source, target = (ROOT/relative).resolve(), (checkout/relative).resolve()
            assert source.is_relative_to(ROOT) and target.is_relative_to(checkout)
            assert source.is_file() and source.suffix != '.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        assert not (checkout/'runs').exists()
        environment = os.environ.copy()
        environment.update({'PYTHONPATH': '', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        for package, script in PACKAGES.items():
            process = subprocess.run([sys.executable, str(checkout/'scripts'/script),
                                      str(checkout/'results'/package)], cwd=checkout,
                                     env=environment, text=True, capture_output=True, timeout=240)
            records.append({'package': package, 'exit_code': process.returncode,
                            'stdout': process.stdout, 'stderr': process.stderr})
            assert process.returncode == 0, records[-1]
            print(f'{package}: clean NumPy-only verification passed', flush=True)
        # Change a reported mean and refresh its checksum. This must fail the
        # independent recalculation, not merely the file-integrity check.
        bundle = checkout/'results/decisive_main_20261122_01'
        path = bundle/'summary.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            data = list(csv.DictReader(handle))
        data[0]['val_loss_mean'] = str(float(data[0]['val_loss_mean'])+.123)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
        manifest_path = bundle/'artifact-manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        row = next(r for r in manifest['files'] if r['path'] == path.relative_to(checkout).as_posix())
        row.update({'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        rejected = subprocess.run([sys.executable, str(checkout/'scripts/verify_decisive_main.py'), str(bundle)],
                                  cwd=checkout, env=environment, text=True, capture_output=True, timeout=240)
        assert rejected.returncode != 0 and 'AssertionError' in rejected.stderr, rejected
        report = {'requires_torch': False, 'torch_installed': False, 'runs_directory_present': False,
                  'copied_manifest_files_including_manifests': len(required),
                  'source_manifest_sha256': {p: hashlib.sha256((ROOT/'results'/p/'artifact-manifest.json').read_bytes()).hexdigest() for p in PACKAGES},
                  'clean_verifications': records,
                  'modified_mean_with_refreshed_checksum_rejected': True,
                  'tamper_rejection_stderr': rejected.stderr[-1800:], 'pass': True}
        destination = ROOT/'reports/decisive_clean_checkout_verification_20261122.json'
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'clean_packages_passed': len(records), 'tamper_rejected': True, 'pass': True}), flush=True)
    finally:
        resolved = checkout.resolve()
        assert resolved.parent == temp_parent and resolved.name.startswith('lorara_decisive_evidence_')
        shutil.rmtree(resolved)


if __name__ == '__main__':
    main()
