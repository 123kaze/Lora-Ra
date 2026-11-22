"""Post-training calibration resource measurement; no optimizer or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from realmodel_experiment import calibrate, evaluate, load_base, tokenize

ROOT = Path(__file__).resolve().parents[1]


def main(run):
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    assert json.loads((run/'completion.json').read_text())['complete']
    identity = json.loads((run/'source_identity.json').read_text())
    source = ROOT/'scripts/realmodel_experiment.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == identity['source_sha256']['scripts/realmodel_experiment.py']
    assets = json.loads((run/'assets.json').read_text())['files']
    prepared = json.loads((run/'data_manifest.json').read_text())['records']
    for record in assets+prepared:
        path = (ROOT/record['path']).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], record['path']
    destination = ROOT/'runs'/(run.name+'_calibration_profile')
    assert not destination.exists(), 'Do not overwrite a resource measurement'
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    base, tokenizer = load_base(cfg)
    original_baselines = json.loads((run/'frozen_baselines.json').read_text())
    baseline_error = 0.
    for task in cfg['tasks']:
        current = evaluate(base, tokenize(cfg, tokenizer, task, 'validation'), cfg)
        baseline_error = max(baseline_error, *(abs(current[key]-original_baselines[task][key]) for key in ('nll', 'accuracy', 'f1')))
    assert baseline_error == 0.
    records = []
    for task in cfg['tasks']:
        data = tokenize(cfg, tokenizer, task, 'train')
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin = time.perf_counter()
        base.float()
        allocation = calibrate(base, data, cfg, destination/task)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        calibration_seconds = time.perf_counter()-begin
        differences = {'G': 0., 'Hin': 0., 'Hout': 0.}
        metadata = json.loads((run/'calibration'/task/'modules.json').read_text())
        for module in metadata.values():
            with np.load(run/'calibration'/task/module['file']) as old, np.load(destination/task/module['file']) as new:
                for key in differences:
                    differences[key] = max(differences[key], float(np.max(np.abs(old[key].astype(np.float64)-new[key]))))
        original = json.loads((run/'calibration'/task/'allocations.json').read_text())['methods']
        records.append({'task': task, 'seconds': calibration_seconds,
            'including_array_recheck_seconds': time.perf_counter()-begin,
            'peak_cuda_allocated_bytes': peak, 'forward_dtype': 'float32',
            'calibration_examples': cfg['calibration_examples'], 'array_recheck_max_abs_error': differences,
            'post_training_resource_probe': True, 'optimizer_steps': 0,
            'same_static_allocations': all(item['ranks'] == original[name]['ranks'] for name, item in allocation['methods'].items())})
        base.bfloat16(); base.requires_grad_(False); torch.cuda.empty_cache()
    output = {'records': records, 'used_for_hyperparameter_selection': False,
              'same_GPU_training_concurrent': False, 'pinned_asset_and_data_files_rehashed': len(assets)+len(prepared),
              'frozen_baseline_max_metric_reload_error': baseline_error,
              'source_sha256': identity['source_sha256']}
    (run/'calibration_cost_profile.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run.resolve())
