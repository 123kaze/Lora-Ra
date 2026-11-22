"""FP32 calibration for explicitly named linear targets, without validation access."""
from __future__ import annotations

import hashlib

import torch
from torch.nn import functional as F

from .data import random_batch
from .experiment import base_digest


def collect_diagonals(model, targets, train, seq_len, batch_size, batches, seed, device):
    model.eval()
    model.requires_grad_(False)
    stats, handles = {}, []
    for target in targets:
        name = target['module']
        linear = model.get_submodule(name)
        linear.weight.requires_grad_(True)
        stats[name] = {'G': torch.zeros_like(linear.weight, dtype=torch.float64, device='cpu'),
                       'Hin': torch.zeros(linear.in_features, dtype=torch.float64),
                       'Hout': torch.zeros(linear.out_features, dtype=torch.float64),
                       'count': 0, 'backward_count': 0}
        def capture(module, inputs, output, key=name):
            x = inputs[0].detach().reshape(-1, module.in_features)
            count = x.shape[0]
            stats[key]['Hin'].add_(x.double().square().sum(0).cpu())
            stats[key]['count'] += count
            def back(gradient):
                delta = gradient.detach().reshape(-1, module.out_features).double() * count
                stats[key]['Hout'].add_(delta.square().sum(0).cpu())
                stats[key]['backward_count'] += count
            output.register_hook(back)
        handles.append(linear.register_forward_hook(capture))
    before = base_digest(model)
    rng = torch.Generator().manual_seed(seed)
    trace, losses = hashlib.sha256(), []
    try:
        for _ in range(batches):
            x, y = random_batch(train, batch_size, seq_len, rng, device)
            trace.update(x.cpu().numpy().tobytes())
            trace.update(y.cpu().numpy().tobytes())
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite calibration loss')
            loss.backward()
            losses.append(loss.item())
            for target in targets:
                name = target['module']
                gradient = model.get_submodule(name).weight.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise ValueError('Missing/nonfinite gradient: ' + name)
                stats[name]['G'].add_(gradient.detach().cpu().double() / batches)
        for value in stats.values():
            assert value['count'] == value['backward_count'] == batches * batch_size * seq_len
            value['Hin'] /= value['count']
            value['Hout'] /= value['count']
        assert base_digest(model) == before
        return stats, {'batch_losses': losses, 'data_trace_sha256': trace.hexdigest(),
                       'predicted_tokens': batches * batch_size * seq_len,
                       'base_unchanged': True, 'optimizer_steps': 0, 'validation_accessed': False,
                       'test_accessed': False, 'precision': 'float32; moment/gradient accumulation float64'}
    finally:
        for handle in handles:
            handle.remove()
