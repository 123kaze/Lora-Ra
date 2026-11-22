"""Exact frozen-prefix reuse when every trainable parameter is in the final block."""
from __future__ import annotations

import hashlib

import torch
from torch.nn import functional as F

from .data import evaluation_batches, random_batch
from .experiment import base_digest


def suffix_forward(model, hidden):
    hidden = model.blocks[-1](hidden)
    return model.lm_head(model.final_norm(hidden))


@torch.no_grad()
def prefix_forward(model, tokens):
    hidden = model.embedding(tokens)
    for block in model.blocks[:-1]:
        hidden = block(hidden)
    return hidden


def assert_only_final_block_trainable(model):
    allowed = f'blocks.{len(model.blocks)-1}.'
    outside = [name for name, p in model.named_parameters() if p.requires_grad and not name.startswith(allowed)]
    if outside:
        raise ValueError('Prefix cache rejected trainable prefix/head parameters: ' + ', '.join(outside))


@torch.no_grad()
def build_training_cache(experiment, base_model, adapter_seed, steps, batch_size, trace_steps):
    base_model.eval()
    before = base_digest(base_model)
    generator = torch.Generator().manual_seed(adapter_seed + 100000)
    cache, targets = None, None
    trace, traces = hashlib.sha256(), {}
    first_tokens = None
    for step in range(1, steps+1):
        x, y = random_batch(experiment.splits['target_train'], batch_size,
                            experiment.model_config.seq_len, generator, experiment.device)
        with experiment.autocast():
            hidden = prefix_forward(base_model, x)
        if cache is None:
            required = steps * (hidden.numel()*hidden.element_size() + y.numel()*y.element_size())
            if experiment.device.type == 'cuda':
                available, _ = torch.cuda.mem_get_info(experiment.device)
                if required + 512*2**20 > available:
                    raise MemoryError('Not enough GPU space for exact prefix cache; reduce unrelated GPU use')
            cache = torch.empty((steps, *hidden.shape), dtype=hidden.dtype, device=hidden.device)
            targets = torch.empty((steps, *y.shape), dtype=y.dtype, device=y.device)
            first_tokens = x.detach().clone()
        cache[step-1].copy_(hidden)
        targets[step-1].copy_(y)
        trace.update(x.cpu().numpy().tobytes())
        trace.update(y.cpu().numpy().tobytes())
        if step in trace_steps:
            traces[str(step)] = trace.hexdigest()
    assert base_digest(base_model) == before
    return {'hidden': cache, 'targets': targets, 'first_tokens': first_tokens,
            'trace_by_step': traces, 'adapter_seed': adapter_seed, 'steps': steps,
            'bytes_allocated': cache.numel()*cache.element_size() + targets.numel()*targets.element_size(),
            'prefix_dtype': str(cache.dtype), 'base_sha256': before}


@torch.no_grad()
def build_validation_cache(experiment, base_model):
    base_model.eval()
    output = []
    cfg = experiment.config['evaluation']
    for x, y in evaluation_batches(experiment.splits['target_val'], cfg['batch_size'],
                                   experiment.model_config.seq_len, cfg['max_blocks'], experiment.device):
        with experiment.autocast():
            hidden = prefix_forward(base_model, x)
        output.append((hidden.detach(), y.detach()))
    return output


@torch.no_grad()
def evaluate_suffix(experiment, model, validation_cache):
    assert_only_final_block_trainable(model)
    model.eval()
    total, correct, count = 0., 0, 0
    for hidden, targets in validation_cache:
        with experiment.autocast():
            logits = suffix_forward(model, hidden)
        total += F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction='sum').item()
        correct += (logits.argmax(-1) == targets).sum().item()
        count += targets.numel()
    return {'loss': total/count, 'accuracy': correct/count, 'bytes': count}
