"""Reload all final label-screen adapters and audit paired data/source identity."""
import argparse
from pathlib import Path

import torch

from carryrank.capacity import dataset
from carryrank.diagnostic import rank_scale
from carryrank.experiment import ROOT, Experiment, base_digest, digest, read_json, write_json
from carryrank.lora import GroupedLinear, attach, merge_all
from carryrank.model import ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = read_json(run / 'config.json')
    identity = read_json(run / 'identity.json')
    for relative, expected in identity['sources'].items():
        assert digest((ROOT / relative).read_bytes()) == expected
        assert digest((run / 'source_snapshot' / relative).read_bytes()) == expected
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    splits, manifest = dataset(identity['parent_training_config'])
    assert manifest == identity['data']
    records, traces = [], {}
    for profile in cfg['widths']:
        for seed in cfg['base_seeds']:
            child = run / f'{profile}_base_{seed}'
            config = read_json(child / 'config.json')
            experiment = object.__new__(Experiment)
            experiment.config, experiment.output, experiment.splits = config, child, splits
            experiment.device, experiment.dtype = torch.device('cuda'), torch.bfloat16
            experiment.model_config = ModelConfig(**config['model'])
            for path in sorted(child.glob('diagnostic_adapters/*/result.json')):
                result = read_json(path)
                final = result['evaluations'][-1]
                saved = torch.load(path.parent / 'adapter.pt', weights_only=True, map_location='cpu')
                model, base = experiment.load_base()
                params = attach(model, saved['groups'], saved['ranks'])
                for module in model.modules():
                    if isinstance(module, GroupedLinear):
                        for factor in module.factors:
                            factor.scaling = rank_scale(saved['mode'], factor.A.shape[0])
                state = model.state_dict()
                assert set(saved['adapter']) == {k for k in state if '.factors.' in k}
                state.update(saved['adapter'])
                model.load_state_dict(state, strict=True)
                assert base_digest(model) == base['sha256'] == saved['base_sha256']
                assert params == cfg['budget'] == final['budget']
                assert result['base_frozen'] and result['zero_initial_error'] == 0
                value = experiment.evaluate(model, 'target_val')
                error = abs(value['loss'] - final['val_loss'])
                assert error < 1e-10 and value['accuracy'] == final['val_accuracy']
                probe = splits['target_train'][:experiment.model_config.seq_len][None].to('cuda')
                with torch.no_grad():
                    original = model(probe)
                merge_all(model)
                with torch.no_grad():
                    merge_error = (model(probe) - original).abs().max().item()
                assert merge_error < 1e-4
                for row in result['evaluations']:
                    traces.setdefault((row['adapter_seed'], row['step']), set()).add(row['data_trace_sha256'])
                records.append({'profile': profile, 'base_seed': seed, 'adapter_seed': final['adapter_seed'],
                                'method': final['method'], 'parameters': params, 'val_reload_error': error,
                                'merge_error': merge_error, 'zero_initial_error': result['zero_initial_error'],
                                'base_frozen': result['base_frozen'],
                                'adapter_file_sha256': digest((path.parent / 'adapter.pt').read_bytes()), 'pass': True})
                del model
    expected = read_json(run / 'completion.json')['trajectories']
    assert len(records) == expected
    assert all(len(t) == 1 for t in traces.values())
    parent = ROOT / cfg['capacity_parent']
    prior = read_json(parent / 'prior_artifacts.json')
    assert all(digest((ROOT / p).read_bytes()) == expected for p, expected in prior.items())
    report = {'checkpoints_reloaded': len(records), 'records': records,
              'max_validation_reload_error': max(r['val_reload_error'] for r in records),
              'max_merge_error': max(r['merge_error'] for r in records), 'exact_budget': cfg['budget'],
              'source_snapshots_verified': len(identity['sources']), 'paired_batch_traces_match': True,
              'prior_artifacts_verified': len(prior), 'prior_artifacts_unchanged': True,
              'test_split_used': False, 'pass': True}
    write_json(run / 'checkpoint_audit.json', report)
    print({k: v for k, v in report.items() if k != 'records'})


if __name__ == '__main__':
    main()
