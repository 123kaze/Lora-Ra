"""Read-only investigation of phase-1 input scale and omitted baseline metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from carryrank.data import prepare, random_batch
from carryrank.experiment import Experiment, base_digest, write_csv, write_json
from carryrank.model import ModelConfig, VariableWidthLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, default=Path('runs/pilot_20261122_01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    parent, output = args.parent.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh probe output directory')
    if not output.is_relative_to(root / 'results'):
        raise ValueError('Probe output must be under results/')
    cfg = json.loads((parent/'config.json').read_text())
    for relative, expected in json.loads((parent/'identity.json').read_text())['code'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Parent implementation changed: {relative}')
    torch.set_num_threads(cfg['runtime']['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    base = torch.load(parent/'base.pt', map_location='cpu', weights_only=True)
    calibration = torch.load(parent/'calibration.pt', map_location='cpu', weights_only=True)
    model = VariableWidthLM(ModelConfig(**cfg['model'])).to(device)
    model.load_state_dict(base['model'])
    assert base_digest(model) == base['sha256'] == calibration['base_sha256']
    splits, manifest = prepare(root, cfg['data'])
    assert manifest == json.loads((parent/'data_manifest.json').read_text())
    second, counts, hooks = {}, {}, []
    for name, linear, _ in model.targets():
        second[name] = torch.zeros(linear.in_features, dtype=torch.float64)
        counts[name] = 0
        def hook(module, inputs, key=name):
            x = inputs[0].detach().reshape(-1, module.in_features)
            second[key].add_(x.double().square().sum(0).cpu())
            counts[key] += x.shape[0]
        hooks.append(linear.register_forward_pre_hook(hook))
    rng = torch.Generator().manual_seed(cfg['calibration']['seed'])
    model.eval()
    with torch.no_grad():
        for _ in range(cfg['calibration']['batches']):
            x, _ = random_batch(splits['target_train'], cfg['calibration']['batch_size'], cfg['model']['seq_len'], rng, device)
            model(x)
    for handle in hooks:
        handle.remove()
    rows = []
    for group in calibration['groups']['true']:
        name, indices = group['module'], group['indices']
        moment = second[name] / counts[name]
        matrix = calibration['gradients'][name][:, indices]
        normalized = matrix / (moment[indices] + 0.01 * moment.mean()).sqrt()
        values = torch.linalg.svdvals(matrix)
        whitened = torch.linalg.svdvals(normalized)
        for k in (1,2,4,8):
            rows.append({'group': group['id'], 'layer': group['layer'], 'projection': name.split('.')[-1],
                         'label': group['label'], 'rank': k, 'm': group['m'], 'n': group['n'],
                         'mean_input_second_moment': moment[indices].mean().item(),
                         'raw_utility': (0.5*values[:k].square().sum()).item(),
                         'diag_input_utility': (0.5*whitened[:k].square().sum()).item()})
    # Borrow only the pure evaluation method, without opening or modifying a run.
    evaluator = object.__new__(Experiment)
    evaluator.config, evaluator.device, evaluator.splits = cfg, device, splits
    evaluator.dtype, evaluator.model_config = torch.bfloat16, ModelConfig(**cfg['model'])
    metrics = {split: evaluator.evaluate(model, split) for split in ('source_val','target_val','target_test')}
    assert base_digest(model) == base['sha256']
    output.mkdir(parents=True)
    write_csv(output/'input_scale_probe.csv', rows)
    write_json(output/'input_second_moments.json', {name:(value/counts[name]).tolist() for name,value in second.items()})
    write_json(output/'frozen_base_metrics.json', {'parent_run':parent.name, 'base_sha256':base['sha256'], 'metrics':metrics,
                                                 'note':'Supplementary evaluation; no retraining or old-run mutation'})
    write_json(output/'provenance.json', {'parent_run':parent.name,'calibration_seed':cfg['calibration']['seed'],
               'calibration_bytes':24576,'input_damping_fraction':0.01,'H_out':'identity',
               'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'base_sha256':base['sha256'],'base_unchanged':True})
    print(json.dumps({'metrics':metrics, 'final_V_rank2':[r for r in rows if r['layer']==4 and r['projection']=='v_proj' and r['rank']==2]},indent=2))


if __name__ == '__main__':
    main()
