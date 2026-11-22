"""Compare full and cached BF16 paths, gradients and fused AdamW on real bases."""
import copy
import hashlib
import json

import torch
from torch.nn import functional as F

from carryrank.axis_lora import attach_axes
from carryrank.capacity import dataset
from carryrank.data import random_batch
from carryrank.experiment import ROOT, read_json, seed_all, write_json
from carryrank.model import ModelConfig, VariableWidthLM
from carryrank.preflight import grouped, target_specifications
from carryrank.prefix_cache import prefix_forward, suffix_forward


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    splits, _ = dataset(read_json(ROOT/'configs/pilot.json'))
    results = []
    for profile in ('long', 'short'):
        parent = ROOT/f'runs/crossdomain_labels_20261122_01/{profile}_base_2026'
        cfg = read_json(parent/'config.json')
        checkpoint = torch.load(parent/'base.pt', weights_only=True, map_location='cpu')
        model = VariableWidthLM(ModelConfig(**cfg['model'])).cuda()
        model.load_state_dict(checkpoint['model'])
        groups = grouped(target_specifications(model, 'matched_final'))
        seed_all(11)
        attach_axes(model, groups, {g['id']: 1 for g in groups})
        other = copy.deepcopy(model)
        rng = torch.Generator().manual_seed(100011)
        x, y = random_batch(splits['target_train'], 16, 128, rng, 'cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = prefix_forward(model, x)
        a = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, fused=True, weight_decay=0.)
        b = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad], lr=.001, fused=True, weight_decay=0.)
        errors = {'logits': 0., 'gradients': 0., 'parameters': 0.}
        for _ in range(3):
            a.zero_grad(); b.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                full, cached = model(x), suffix_forward(other, hidden)
                la = F.cross_entropy(full.float().flatten(0, 1), y.flatten())
                lb = F.cross_entropy(cached.float().flatten(0, 1), y.flatten())
            errors['logits'] = max(errors['logits'], (full-cached).abs().max().item())
            la.backward(); lb.backward()
            for p, q in zip(model.parameters(), other.parameters()):
                if p.requires_grad:
                    errors['gradients'] = max(errors['gradients'], (p.grad-q.grad).abs().max().item())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            torch.nn.utils.clip_grad_norm_([p for p in other.parameters() if p.requires_grad], 1.)
            a.step(); b.step()
            for p, q in zip(model.parameters(), other.parameters()):
                errors['parameters'] = max(errors['parameters'], (p-q).abs().max().item())
        assert all(v == 0 for v in errors.values()), errors
        results.append({'profile': profile, 'base_seed': 2026, 'optimizer_steps_for_test': 3,
                        'not_research_data': True, **errors, 'pass': True})
        del model, other, a, b, hidden
    report = {'records': results, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'prefix_cache_source_sha256': hashlib.sha256((ROOT/'src/carryrank/prefix_cache.py').read_bytes()).hexdigest(),
              'axis_lora_source_sha256': hashlib.sha256((ROOT/'src/carryrank/axis_lora.py').read_bytes()).hexdigest(),
              'pass': True}
    write_json(ROOT/'reports/decisive_prefix_equivalence.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
