"""Prospective cross-domain capacity screen, separate from method comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import digest, prepare, random_batch
from .experiment import (ROOT, Experiment, base_digest, log, new_attempt, read_json,
                         seed_all, synchronize, write_csv, write_json)
from .lora import GroupedLinear, attach, merge_all
from .model import ModelConfig, structure


def sources():
    paths = sorted((ROOT / 'src').rglob('*.py'))
    paths += [ROOT / 'scripts/prepare_wikitext.py']
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in paths}


def dataset(config):
    splits, source = prepare(ROOT, config['data'])
    if source['raw_sha256'] != '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed':
        raise ValueError('Source corpus differs')
    manifest = read_json(ROOT / 'data/processed/wikitext-2-raw-v1/manifest.json')
    if manifest['revision'] != 'b08601e04326c79dfdd32d625aee71d232d685c3':
        raise ValueError('Unexpected target revision')
    # Explicitly exclude all old target splits and the WikiText test split.
    splits = {k: splits[k] for k in ('source_train', 'source_val')}
    for record in manifest['splits']:
        value = (ROOT / record['bin_path']).read_bytes()
        if len(value) != record['bin_bytes'] or digest(value) != record['bin_sha256']:
            raise ValueError('Target corpus cache hash mismatch')
        name = {'train': 'target_train', 'validation': 'target_val'}[record['split']]
        splits[name] = torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy()).long()
    return splits, {'source': source, 'target': manifest}


def final_v_groups(widths):
    rec = structure(widths)[-1]
    name = f"blocks.{len(widths)-1}.attn.v_proj"
    groups = [{'id': name + ':' + label, 'module': name, 'label': label,
               'indices': rec[label], 'm': rec['width'], 'n': len(rec[label]),
               'cost': rec['width'] + len(rec[label])} for label in ('C', 'R')]
    if not groups[0]['n'] == groups[1]['n'] == 64:
        raise ValueError('This protocol requires equal 64-column C/R groups')
    return groups


def capacity_gate(rows, config):
    if config.get('smoke'):
        return {'status': 'smoke_ineligible', 'pass': False}
    specification = config['gate']
    outcomes = []
    for profile in config['widths']:
        for base_seed in config['base_seeds']:
            differences = []
            for seed in config['adapter_seeds']:
                pair = {int(r['rank']): float(r['val_loss']) for r in rows
                        if r['profile'] == profile and int(r['base_seed']) == base_seed
                        and int(r['adapter_seed']) == seed and int(r['step']) == specification['step']}
                differences.append(pair[specification['low_rank']] - pair[specification['high_rank']])
            mean = statistics.mean(differences)
            outcomes.append({'profile': profile, 'base_seed': base_seed,
                'paired_rank1_minus_rank4': differences, 'mean': mean,
                'sample_std': statistics.stdev(differences),
                'pass': mean >= specification['minimum_mean_loss_reduction'] and all(x > 0 for x in differences)})
    return {'status': 'capacity_screen_only', 'pass': all(x['pass'] for x in outcomes),
            'specification': specification, 'outcomes': outcomes,
            'independent_base_seeds_per_profile': len(config['base_seeds']),
            'method_advantage_tested': False, 'test_split_used': False}


def train(experiment, cfg, profile, base_seed, adapter_seed, rank, resume):
    unit = experiment.output / 'adapters' / f'rank{rank}_seed{adapter_seed}'
    if (unit / 'result.json').exists():
        if not resume:
            raise FileExistsError(unit)
        result = read_json(unit / 'result.json')
        if digest((unit / 'adapter.pt').read_bytes()) != result['adapter_file_sha256']:
            raise ValueError('Existing adapter file hash mismatch')
        return result
    unit.mkdir(parents=True, exist_ok=True)
    model, base = experiment.load_base()
    groups = final_v_groups(experiment.model_config.widths)
    ranks = {g['id']: rank for g in groups}
    probe = experiment.splits['target_train'][:experiment.model_config.seq_len][None].to(experiment.device)
    model.eval()
    with torch.no_grad():
        original = model(probe)
    seed_all(adapter_seed)
    count = attach(model, groups, ranks, 1 / math.sqrt(rank))
    expected = sum(g['cost'] * rank for g in groups)
    assert count == expected
    with torch.no_grad():
        initial_error = (model(probe) - original).abs().max().item()
    assert initial_error == 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=0.)
    generator = torch.Generator().manual_seed(adapter_seed + 100000)
    trace = hashlib.sha256()
    evaluations = []
    synchronize(experiment.device)
    torch.cuda.reset_peak_memory_stats(experiment.device)
    started = time.perf_counter()
    key = f'{profile}_base{base_seed}_rank{rank}_seed{adapter_seed}'
    with new_attempt(experiment.output, key) as handle:
        for step in range(1, cfg['steps'] + 1):
            model.train()
            x, y = random_batch(experiment.splits['target_train'], experiment.config['adapter']['batch_size'],
                                experiment.model_config.seq_len, generator, experiment.device)
            trace.update(x.cpu().numpy().tobytes())
            trace.update(y.cpu().numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            with experiment.autocast():
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(key)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            if step == 1 or step % 120 == 0 or step == cfg['steps']:
                log(handle, {'stage': 'capacity_train', 'key': key, 'step': step,
                             'batch_loss': loss.item(), 'elapsed_s': time.perf_counter() - started})
            if step in cfg['eval_steps']:
                validation = experiment.evaluate(model, 'target_val')
                train_metric = experiment.evaluate(model, 'target_train')
                row = {'profile': profile, 'base_seed': base_seed, 'adapter_seed': adapter_seed,
                       'rank': rank, 'budget': count, 'step': step, 'lr': cfg['lr'],
                       'scaling': 1 / math.sqrt(rank), 'val_loss': validation['loss'],
                       'train_eval_loss': train_metric['loss'], 'val_accuracy': validation['accuracy'],
                       'val_bytes': validation['bytes'], 'base_val_loss': base['summary']['target_val']['loss'],
                       'val_improvement': base['summary']['target_val']['loss'] - validation['loss'],
                       'data_trace_sha256': trace.hexdigest(), 'base_sha256': base['sha256']}
                evaluations.append(row)
                log(handle, {'stage': 'capacity_validation', **row})
        synchronize(experiment.device)
        frozen = base_digest(model) == base['sha256']
        assert frozen
        elapsed = time.perf_counter() - started
        peak_memory = experiment.peak_memory()
        model.eval()
        with torch.no_grad():
            before_merge = model(probe)
        checkpoint = {'adapter': {k: v.detach().cpu() for k, v in model.state_dict().items() if '.factors.' in k},
                      'groups': groups, 'ranks': ranks, 'scaling': 1 / math.sqrt(rank),
                      'base_sha256': base['sha256']}
        torch.save(checkpoint, unit / 'adapter.pt')
        merge_all(model)
        with torch.no_grad():
            merge_error = (model(probe) - before_merge).abs().max().item()
        assert merge_error < 1e-4
        result = {'evaluations': evaluations, 'groups': groups, 'ranks': ranks,
                  'zero_initial_error': initial_error, 'base_frozen': frozen,
                  'merge_max_error': merge_error, 'time_s': elapsed, 'peak_allocated_mb': peak_memory,
                  'adapter_file_sha256': digest((unit / 'adapter.pt').read_bytes())}
        write_json(unit / 'result.json', result)
        log(handle, {'stage': 'capacity_complete', 'key': key, 'time_s': elapsed,
                     'merge_max_error': merge_error, 'base_frozen': frozen})
    del model, optimizer, parameters
    torch.cuda.empty_cache()
    return result


def run(config, output, resume=False):
    if output.parent != ROOT / 'runs':
        raise ValueError('Choose a direct runs/<run-id> directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    parent = read_json(ROOT / 'configs/pilot.json')
    splits, data = dataset(parent)
    identity = {'config': config, 'parent_training_config': parent, 'sources': sources(), 'data': data,
                'protocol_sha256': digest((ROOT / 'plan/crossdomain-capacity-protocol.md').read_bytes()),
                'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                'gpu': torch.cuda.get_device_name(0)}
    if output.exists():
        if not resume or read_json(output / 'identity.json') != identity:
            raise ValueError('Resume requires matching config/code/data/environment identity')
    else:
        if resume:
            raise FileNotFoundError(output)
        output.mkdir(parents=True)
        write_json(output / 'identity.json', identity)
        write_json(output / 'config.json', config)
        write_json(output / 'data_manifest.json', data)
        write_json(output / 'environment.json', {'executable': sys.executable, 'platform': platform.platform(),
                    'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                    'precision': 'BF16 forward / FP32 CE and weights', 'bitwise_rerun_guaranteed': False})
        shutil.copy2(ROOT / 'plan/crossdomain-capacity-protocol.md', output / 'protocol.md')
        for relative in identity['sources']:
            destination = output / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        old_paths = [ROOT / '研究方案.md']
        for directory in ('results/pilot_20261122_01', 'results/diagnostic_20261122_02',
                          'results/outputdiag_calibration_20261122_01'):
            old_paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
        write_json(output / 'prior_artifacts.json', {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in old_paths})
    rows, units = [], []
    for profile, widths in config['widths'].items():
        for seed in config['base_seeds']:
            child = output / f'{profile}_base_{seed}'
            child.mkdir(exist_ok=True)
            cfg = copy.deepcopy(parent)
            cfg['name'] = f'capacity_{profile}_base_{seed}'
            cfg['model']['widths'] = widths
            cfg['pretrain']['seed'] = seed
            cfg['evaluation']['max_blocks'] = config['evaluation_max_blocks']
            cfg['runtime']['log_every'] = 120
            if config.get('smoke'):
                cfg['name'] += '_smoke'
                cfg['pretrain'].update(steps=4, eval_every=4, batch_size=2)
                cfg['adapter']['batch_size'] = 2
            write_json(child / 'config.json', cfg)
            write_json(child / 'structure.json', structure(widths))
            experiment = object.__new__(Experiment)
            experiment.config, experiment.output, experiment.device = cfg, child, torch.device('cuda')
            experiment.model_config = ModelConfig(**cfg['model'])
            experiment.dtype, experiment.splits = torch.bfloat16, splits
            experiment.pretrain()
            model, base = experiment.load_base()
            assert sum(p.numel() for p in model.parameters()) == 935424
            if not config.get('smoke') and not base['summary']['pretraining_gate']:
                raise RuntimeError('Pretraining quality gate failed')
            baseline = {split: experiment.evaluate(model, split) for split in ('source_val', 'target_train', 'target_val')}
            write_json(child / 'baseline.json', baseline)
            del model
            for adapter_seed in config['adapter_seeds']:
                for rank in config['ranks']:
                    result = train(experiment, config, profile, seed, adapter_seed, rank, resume)
                    rows.extend(result['evaluations'])
                    units.append({'profile': profile, 'base_seed': seed, 'adapter_seed': adapter_seed,
                                  'rank': rank, **{k: result[k] for k in ('zero_initial_error', 'base_frozen',
                                  'merge_max_error', 'time_s', 'peak_allocated_mb', 'adapter_file_sha256')}})
                    write_csv(output / 'main.csv', rows)
                    write_csv(output / 'trajectories.csv', units)
    write_json(output / 'capacity_gate.json', capacity_gate(rows, config))
    checks = {'expected_trajectories': len(config['widths']) * len(config['base_seeds']) * len(config['adapter_seeds']) * len(config['ranks']),
              'trajectories': len(units), 'rows': len(rows), 'all_base_frozen': all(u['base_frozen'] for u in units),
              'max_zero_initial_error': max(u['zero_initial_error'] for u in units),
              'max_merge_error': max(u['merge_max_error'] for u in units)}
    paired = {}
    for row in rows:
        key = (row['adapter_seed'], row['step'])
        paired.setdefault(key, set()).add(row['data_trace_sha256'])
    checks['paired_batches_across_profiles_and_ranks'] = all(len(v) == 1 for v in paired.values())
    previous = read_json(output / 'prior_artifacts.json')
    checks['prior_artifact_count'] = len(previous)
    checks['prior_artifacts_unchanged'] = all(digest((ROOT / p).read_bytes()) == sha for p, sha in previous.items())
    checks['pass'] = (checks['trajectories'] == checks['expected_trajectories'] and
                      checks['rows'] == len(units) * len(config['eval_steps']) and checks['all_base_frozen'] and
                      checks['max_zero_initial_error'] == 0 and checks['max_merge_error'] < 1e-4 and
                      checks['paired_batches_across_profiles_and_ranks'] and checks['prior_artifacts_unchanged'])
    write_json(output / 'verification.json', checks)
    if not checks['pass']:
        raise AssertionError(checks)
    print({'stage': 'capacity_screen_finished', 'gate': read_json(output / 'capacity_gate.json'), 'verification': checks}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/crossdomain_capacity.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.smoke:
        cfg.update(smoke=True, adapter_seeds=[11], ranks=[1, 4], steps=3, eval_steps=[3], evaluation_max_blocks=4)
    run(cfg, args.output.resolve(), args.resume)


if __name__ == '__main__':
    main()
