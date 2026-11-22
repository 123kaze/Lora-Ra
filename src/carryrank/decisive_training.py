"""Exact-budget adapters with audited frozen-prefix reuse; no method-specific tuning."""
from __future__ import annotations

import copy
import hashlib
import json
import time

import torch
from torch.nn import functional as F

from .axis_lora import attach_axes, merge_axes
from .experiment import (ROOT, Experiment, base_digest, digest, log, new_attempt,
                         read_json, seed_all, synchronize, write_json)
from .model import ModelConfig
from .prefix_cache import assert_only_final_block_trainable, evaluate_suffix, suffix_forward


def create_experiment(config, output, splits):
    experiment = object.__new__(Experiment)
    experiment.config, experiment.output, experiment.splits = config, output, splits
    experiment.device, experiment.dtype = torch.device('cuda'), torch.bfloat16
    experiment.model_config = ModelConfig(**config['model'])
    return experiment


def configuration_key(groups, ranks, alpha):
    active = [{'module': g['module'], 'axis': g.get('axis', 'column'),
               'indices': g.get('indices', list(range(g['n']))), 'rank': ranks.get(g['id'], 0)}
              for g in groups if ranks.get(g['id'], 0)]
    return digest(json.dumps({'active_in_initialization_order': active, 'alpha': alpha}, sort_keys=True).encode())


def train_adapter(experiment, settings, profile, base_seed, adapter_seed, budget, method,
                  groups, ranks, train_cache, val_cache, aliases, resume=False):
    key = f'B{budget}_{method}_seed{adapter_seed}'
    unit = experiment.output / 'adapters' / key
    result_path = unit / 'result.json'
    fingerprint = configuration_key(groups, ranks, settings['alpha'])
    if result_path.exists():
        if not resume:
            raise FileExistsError(unit)
        result = read_json(result_path)
        if result['configuration_sha256'] != fingerprint:
            raise ValueError('Completed adapter configuration differs')
        if not result.get('reused_from'):
            assert digest((unit / 'adapter.pt').read_bytes()) == result['adapter_file_sha256']
            aliases.setdefault(fingerprint, (result, result_path))
        return result
    unit.mkdir(parents=True, exist_ok=True)
    if fingerprint in aliases:
        source, source_path = aliases[fingerprint]
        result = copy.deepcopy(source)
        result['reused_from'] = source_path.relative_to(ROOT).as_posix()
        result['groups'], result['ranks'] = groups, ranks
        for row in result['evaluations']:
            row['method'] = method
            row['reused_from'] = result['reused_from']
        write_json(result_path, result)
        return result
    model, base = experiment.load_base()
    model.eval()
    probe = train_cache['first_tokens'][:1]
    with torch.no_grad():
        original = model(probe)
    seed_all(adapter_seed)
    count = attach_axes(model, groups, ranks, settings['alpha'])
    assert count == budget
    assert_only_final_block_trainable(model)
    with torch.no_grad():
        init_error = (model(probe)-original).abs().max().item()
    assert init_error == 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['lr'], betas=tuple(settings['betas']),
                                  weight_decay=settings['weight_decay'], fused=True)
    evaluations = []
    synchronize(experiment.device)
    torch.cuda.reset_peak_memory_stats(experiment.device)
    started = time.perf_counter()
    interval_loss = torch.zeros((), device=experiment.device)
    interval_count = 0
    cache_error = 0.
    with new_attempt(experiment.output, key) as handle:
        for step in range(1, settings['steps']+1):
            model.train()
            hidden, target = train_cache['hidden'][step-1], train_cache['targets'][step-1]
            optimizer.zero_grad(set_to_none=True)
            with experiment.autocast():
                logits = suffix_forward(model, hidden)
                loss = F.cross_entropy(logits.float().flatten(0, 1), target.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(key)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            interval_loss.add_(loss.detach())
            interval_count += 1
            if step == 1 or step == settings['steps']:
                model.eval()
                with torch.no_grad(), experiment.autocast():
                    full = model(train_cache['first_tokens'])
                    cached = suffix_forward(model, train_cache['hidden'][0])
                cache_error = max(cache_error, (full-cached).abs().max().item())
                assert cache_error == 0, ('Prefix path mismatch', key, cache_error)
            if step == 1 or step % 120 == 0 or step == settings['steps']:
                log(handle, {'stage': 'decisive_train', 'profile': profile, 'base_seed': base_seed,
                             'method': method, 'budget': budget, 'adapter_seed': adapter_seed, 'step': step,
                             'interval_train_loss': interval_loss.item()/interval_count,
                             'elapsed_s': time.perf_counter()-started})
                interval_loss.zero_(); interval_count = 0
            if step in settings['eval_steps']:
                value = evaluate_suffix(experiment, model, val_cache)
                row = {'profile': profile, 'base_seed': base_seed, 'adapter_seed': adapter_seed,
                       'budget': budget, 'method': method, 'step': step, 'parameters': count,
                       'val_loss': value['loss'], 'val_accuracy': value['accuracy'], 'val_bytes': value['bytes'],
                       'base_val_loss': base['summary']['target_val']['loss'],
                       'improvement': base['summary']['target_val']['loss']-value['loss'],
                       'data_trace_sha256': train_cache['trace_by_step'][str(step)],
                       'base_sha256': base['sha256'], 'reused_from': ''}
                evaluations.append(row)
                log(handle, {'stage': 'decisive_validation', **row})
        synchronize(experiment.device)
        frozen = base_digest(model) == base['sha256']
        assert frozen
        elapsed = time.perf_counter()-started
        peak = experiment.peak_memory()
        model.eval()
        with torch.no_grad():
            before_merge = model(probe)
        torch.save({'adapter': {k: v.detach().cpu() for k, v in model.state_dict().items() if '.factors.' in k},
                    'groups': groups, 'ranks': ranks, 'alpha': settings['alpha'],
                    'base_sha256': base['sha256']}, unit/'adapter.pt')
        merge_axes(model)
        with torch.no_grad():
            merge_error = (model(probe)-before_merge).abs().max().item()
        assert merge_error < 1e-4
        result = {'evaluations': evaluations, 'groups': groups, 'ranks': ranks,
                  'configuration_sha256': fingerprint, 'reused_from': '',
                  'zero_initial_error': init_error, 'base_frozen': frozen,
                  'cache_full_max_error': cache_error, 'merge_max_error': merge_error,
                  'time_s': elapsed, 'peak_allocated_mb_including_cache': peak,
                  'adapter_file_sha256': digest((unit/'adapter.pt').read_bytes())}
        log(handle, {'stage': 'decisive_complete', 'key': key, 'time_s': elapsed,
                     'cache_full_max_error': cache_error, 'merge_max_error': merge_error})
        result['log_path'] = str(handle.name)
        write_json(result_path, result)
    aliases[fingerprint] = result, result_path
    del model, optimizer, parameters
    return result
