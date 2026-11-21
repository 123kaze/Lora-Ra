"""Independently reload each completed capacity adapter and repeat final validation."""
import argparse
import math
from pathlib import Path

import torch

from carryrank.capacity import dataset
from carryrank.experiment import ROOT, Experiment, base_digest, digest, read_json, write_json
from carryrank.lora import attach, merge_all
from carryrank.model import ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = read_json(run / 'config.json')
    identity = read_json(run / 'identity.json')
    for name, expected in identity['sources'].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError('Training code changed: ' + name)
        if digest((run / 'source_snapshot' / name).read_bytes()) != expected:
            raise ValueError('Source snapshot differs: ' + name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    splits, manifest = dataset(identity['parent_training_config'])
    assert manifest == identity['data']
    records = []
    for profile in config['widths']:
        for seed in config['base_seeds']:
            child = run / f'{profile}_base_{seed}'
            cfg = read_json(child / 'config.json')
            experiment = object.__new__(Experiment)
            experiment.config, experiment.output = cfg, child
            experiment.device, experiment.dtype = torch.device('cuda'), torch.bfloat16
            experiment.splits, experiment.model_config = splits, ModelConfig(**cfg['model'])
            for adapter_seed in config['adapter_seeds']:
                for rank in config['ranks']:
                    unit = child / 'adapters' / f'rank{rank}_seed{adapter_seed}'
                    result = read_json(unit / 'result.json')
                    assert digest((unit / 'adapter.pt').read_bytes()) == result['adapter_file_sha256']
                    checkpoint = torch.load(unit / 'adapter.pt', weights_only=True, map_location='cpu')
                    model, base = experiment.load_base()
                    assert checkpoint['base_sha256'] == base['sha256']
                    params = attach(model, checkpoint['groups'], checkpoint['ranks'], checkpoint['scaling'])
                    assert params == sum(g['cost'] * rank for g in checkpoint['groups'])
                    missing, unexpected = model.load_state_dict(checkpoint['adapter'], strict=False)
                    assert not unexpected and not any('.factors.' in x for x in missing)
                    assert base_digest(model) == base['sha256']
                    value = experiment.evaluate(model, 'target_val')
                    original = result['evaluations'][-1]
                    error = abs(value['loss'] - original['val_loss'])
                    assert error < 1e-10 and value['accuracy'] == original['val_accuracy']
                    probe = splits['target_train'][:experiment.model_config.seq_len][None].to('cuda')
                    with torch.no_grad():
                        before = model(probe)
                    merge_all(model)
                    with torch.no_grad():
                        merge_error = (model(probe) - before).abs().max().item()
                    assert merge_error < 1e-4
                    records.append({'profile': profile, 'base_seed': seed, 'adapter_seed': adapter_seed,
                                    'rank': rank, 'parameters': params, 'validation_reload_error': error,
                                    'merge_error': merge_error, 'pass': True})
                    del model
    previous = read_json(run / 'prior_artifacts.json')
    prior_unchanged = all(digest((ROOT / name).read_bytes()) == value for name, value in previous.items())
    assert prior_unchanged
    report = {'checkpoints_reloaded': len(records), 'records': records,
              'max_validation_reload_error': max(r['validation_reload_error'] for r in records),
              'max_merge_error': max(r['merge_error'] for r in records),
              'prior_artifacts_verified': len(previous), 'prior_artifacts_unchanged': prior_unchanged,
              'test_split_used': False, 'pass': all(r['pass'] for r in records)}
    write_json(run / 'checkpoint_audit.json', report)
    print({k: v for k, v in report.items() if k != 'records'})


if __name__ == '__main__':
    main()
