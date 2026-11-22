"""Incrementally reload completed stage-0 adapters using full, uncached models."""
import argparse
from pathlib import Path

import torch

from carryrank.axis_lora import attach_axes, merge_axes
from carryrank.capacity import dataset
from carryrank.decisive_training import configuration_key, create_experiment
from carryrank.experiment import ROOT, base_digest, digest, read_json, write_json


def audit(run, allow_incomplete=False):
    run = run.resolve()
    identity = read_json(run/'identity.json')
    cfg = read_json(run/'config.json')
    for name, expected in identity['source'].items():
        assert digest((ROOT/name).read_bytes()) == expected, name
        assert digest((run/'source_snapshot'/name).read_bytes()) == expected, name
    complete = (run/'completion.json').exists()
    if not complete and not allow_incomplete:
        raise ValueError('Run is incomplete; use --allow-incomplete only for an incremental audit')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    splits, data = dataset(read_json(ROOT/'configs/pilot.json'))
    assert data == identity['data']
    old = read_json(run/'checkpoint_audit.json') if (run/'checkpoint_audit.json').exists() else {'records': []}
    done = {r['result_path']: r for r in old['records']}
    records, logical, new_checks = [], 0, 0
    traces = {}
    for child in sorted(run.glob('*_base_*')):
        exp = create_experiment(read_json(child/'config.json'), child, splits)
        for path in sorted(child.glob('adapters/*/result.json')):
            result = read_json(path)
            logical += 1
            source_path = path
            if result['reused_from']:
                source_path = (ROOT/result['reused_from']).resolve()
                assert source_path.is_relative_to(run)
                source = read_json(source_path)
                assert not source['reused_from']
                assert result['configuration_sha256'] == source['configuration_sha256']
                assert configuration_key(result['groups'], result['ranks'], cfg['alpha']) == source['configuration_sha256']
                for a, b in zip(result['evaluations'], source['evaluations']):
                    for field in ('profile', 'base_seed', 'adapter_seed', 'budget', 'step', 'val_loss',
                                  'val_accuracy', 'val_bytes', 'data_trace_sha256', 'base_sha256'):
                        assert a[field] == b[field]
            else:
                key = path.relative_to(run).as_posix()
                tensor = path.parent/'adapter.pt'
                actual_sha = digest(tensor.read_bytes())
                assert actual_sha == result['adapter_file_sha256']
                if key in done:
                    assert done[key]['adapter_file_sha256'] == actual_sha
                    records.append(done[key])
                else:
                    saved = torch.load(tensor, map_location='cpu', weights_only=True)
                    model, base = exp.load_base()
                    count = attach_axes(model, saved['groups'], saved['ranks'], saved['alpha'])
                    state = model.state_dict()
                    assert set(saved['adapter']) == {k for k in state if '.factors.' in k}
                    state.update(saved['adapter']); model.load_state_dict(state, strict=True)
                    assert base_digest(model) == base['sha256'] == saved['base_sha256']
                    value = exp.evaluate(model, 'target_val')
                    final = result['evaluations'][-1]
                    error = abs(value['loss']-final['val_loss'])
                    assert error == 0 and value['accuracy'] == final['val_accuracy'], (path, error)
                    assert count == final['budget']
                    probe = splits['target_train'][:exp.model_config.seq_len][None].cuda()
                    with torch.no_grad():
                        original = model(probe)
                    merge_axes(model)
                    with torch.no_grad():
                        merge_error = (model(probe)-original).abs().max().item()
                    assert merge_error < 1e-4
                    records.append({'result_path': key, 'adapter_file_sha256': actual_sha,
                        'profile': final['profile'], 'base_seed': final['base_seed'],
                        'adapter_seed': final['adapter_seed'], 'budget': count, 'method': final['method'],
                        'validation_reload_error': error, 'merge_error': merge_error, 'pass': True})
                    new_checks += 1
                    del model
            for row in result['evaluations']:
                traces.setdefault((row['adapter_seed'], row['step']), set()).add(row['data_trace_sha256'])
    assert all(len(v) == 1 for v in traces.values())
    if complete:
        expected = read_json(run/'completion.json')
        assert logical == expected['logical_conditions']
        assert len(records) == expected['actual_optimizer_trajectories']
    prior = read_json(run/'prior_artifacts.json')
    assert all(digest((ROOT/p).read_bytes()) == expected for p, expected in prior.items())
    report = {'records': records, 'logical_conditions_verified': logical,
        'actual_checkpoints_reloaded': len(records), 'new_checkpoints_this_invocation': new_checks,
        'max_validation_reload_error': max([r['validation_reload_error'] for r in records] or [0.]),
        'max_merge_error': max([r['merge_error'] for r in records] or [0.]),
        'paired_data_traces_match': True, 'old_artifacts_verified': len(prior),
        'old_artifacts_unchanged': True, 'complete_run': complete, 'pass': True}
    write_json(run/'checkpoint_audit.json', report)
    print({k: v for k, v in report.items() if k != 'records'}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    audit(args.run, args.allow_incomplete)
