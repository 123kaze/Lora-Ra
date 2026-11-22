"""Pinned GPT-2 classification probe: calibration, LR selection, training and reload audit."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/realmodel_gpt2.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def seed_all(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parameter_hash(parameters):
    trace = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        trace.update(name.encode())
        trace.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return trace.hexdigest()


def identity(cfg):
    names = ['scripts/realmodel_experiment.py', 'scripts/prepare_realmodel_assets.py',
             'scripts/prepare_realmodel_data.py', 'configs/realmodel_gpt2.json', cfg['protocol']]
    return {name: digest(ROOT/name) for name in names}


def initialize(run, cfg, smoke):
    import accelerate, peft, transformers
    sources = identity(cfg)
    current = {'source_sha256': sources, 'assets_sha256': digest(ROOT/cfg['assets']/'assets.json'),
               'data_manifest_sha256': digest(ROOT/cfg['data']/'manifest.json'),
               'smoke': smoke, 'python': sys.version, 'torch': torch.__version__,
               'transformers': transformers.__version__, 'peft': peft.__version__,
               'accelerate': accelerate.__version__, 'numpy': np.__version__,
               'gpu': torch.cuda.get_device_name(), 'platform': platform.platform()}
    if (run/'source_identity.json').exists():
        assert read(run/'source_identity.json') == current, 'Run identity changed; use a new run ID'
    else:
        run.mkdir(parents=True, exist_ok=True)
        write(run/'source_identity.json', current); write(run/'config.json', cfg)
        shutil.copy2(ROOT/cfg['protocol'], run/'protocol.md')
        shutil.copy2(ROOT/cfg['assets']/'assets.json', run/'assets.json')
        shutil.copy2(ROOT/cfg['data']/'manifest.json', run/'data_manifest.json')
        for name in names_from(sources):
            target = run/'source_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
        old = [p for p in (ROOT/'results').rglob('*') if p.is_file()]
        old += [ROOT/'研究方案.md']
        write(run/'prior_artifacts.json', [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)} for p in old])


def names_from(mapping):
    return list(mapping)


def load_base(cfg):
    path = ROOT/cfg['assets']/'model'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    base.config.use_cache = False
    for module in base.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    base.requires_grad_(False); base.eval()
    return base, tokenizer


def tokenize(cfg, tokenizer, task, split):
    rows = [json.loads(line) for line in (ROOT/cfg['data']/f'{task}_{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    words = [' negative', ' positive'] if task == 'sst2' else [' no', ' yes']
    candidates = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    assert all(len(ids) == 1 for ids in candidates), (task, words, candidates)
    candidates = [ids[0] for ids in candidates]
    ids, masks, labels, example_ids, truncated = [], [], [], [], 0
    for row in rows:
        if task == 'sst2':
            prompt = f"Review: {row['sentence']}\nSentiment:"
        else:
            prompt = f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}\nParaphrase:"
        value = tokenizer.encode(prompt, add_special_tokens=False)
        truncated += len(value) > cfg['sequence_length']
        value = value[-cfg['sequence_length']:]
        padding = cfg['sequence_length']-len(value)
        ids.append([tokenizer.eos_token_id]*padding+value)
        masks.append([0]*padding+[1]*len(value))
        labels.append(int(row['label'])); example_ids.append(int(row['idx']))
    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(masks, dtype=torch.long)
    position = (mask.cumsum(-1)-1).clamp_min(0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {'input_ids': ids, 'attention_mask': mask, 'position_ids': position,
            'targets': torch.tensor(candidates)[labels], 'labels': labels, 'ids': example_ids,
            'candidate_ids': candidates, 'truncated': int(truncated)}


def batch(data, indices):
    return {key: data[key][indices].to('cuda') for key in ('input_ids', 'attention_mask', 'position_ids')}


def forward(model, inputs, targets=None):
    options = {} if targets is None else {'labels': targets[:, None], 'shift_labels': targets[:, None]}
    return model(**inputs, logits_to_keep=1, use_cache=False, **options)


@torch.no_grad()
def evaluate(model, data, cfg, destination=None):
    model.eval(); output = []
    for start in range(0, len(data['labels']), cfg['eval_batch']):
        indices = list(range(start, min(start+cfg['eval_batch'], len(data['labels']))))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = forward(model, batch(data, indices)).logits[:, -1].float()
        logprob = logits.log_softmax(-1)
        label_logits = logits[:, data['candidate_ids']].cpu()
        targets = data['targets'][indices].to('cuda')
        correct_logprob = logprob.gather(1, targets[:, None]).squeeze(1).cpu()
        predictions = label_logits.argmax(-1)
        for i, index in enumerate(indices):
            output.append({'example_id': data['ids'][index], 'label': int(data['labels'][index]),
                'prediction': int(predictions[i]), 'target_logprob': float(correct_logprob[i]),
                'label0_logit': float(label_logits[i, 0]), 'label1_logit': float(label_logits[i, 1])})
    labels = np.array([r['label'] for r in output]); pred = np.array([r['prediction'] for r in output])
    tp = int(np.sum((labels == 1)&(pred == 1))); fp = int(np.sum((labels == 0)&(pred == 1))); fn = int(np.sum((labels == 1)&(pred == 0)))
    metrics = {'nll': -float(np.mean([r['target_logprob'] for r in output])),
               'accuracy': float(np.mean(labels == pred)), 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
               'examples': len(output)}
    if destination:
        csv_write(destination, output)
    return metrics


def solve_rank(modules, utility, budget, cap, floor=0):
    states = {0: (0., [])}
    for name in modules:
        fresh = {}
        for used, (score, values) in states.items():
            for rank in range(floor, cap+1):
                target = used+rank
                if target > budget:
                    continue
                candidate = score+float(utility[name][rank])
                if target not in fresh or candidate > fresh[target][0]+1e-14:
                    fresh[target] = (candidate, values+[rank])
        states = fresh
    objective, values = states[budget]
    return dict(zip(modules, values)), objective


def calibrate(base, data, cfg, folder):
    if (folder/'allocations.json').exists():
        return read(folder/'allocations.json')
    folder.mkdir(parents=True, exist_ok=True)
    modules = {name: module for name, module in base.named_modules() if name.endswith(cfg['target_suffix'])}
    assert len(modules) == 12 and all(tuple(m.weight.shape) == (768, 2304) for m in modules.values())
    stats = {name: {'Hin': torch.zeros(768, dtype=torch.float64, device='cuda'),
                    'Hout': torch.zeros(2304, dtype=torch.float64, device='cuda'), 'tokens': 0} for name in modules}
    handles, context = [], {}
    for name, module in modules.items():
        module.weight.requires_grad_(True)
        def hook(_module, inputs, output, name=name):
            value = inputs[0].detach().double()
            mask, label_count = context['mask'], context['labels']
            stats[name]['Hin'] += (value.square()*mask[..., None]).sum((0, 1))
            stats[name]['tokens'] += int(mask.sum())
            def backward(gradient):
                stats[name]['Hout'] += ((gradient.detach().double()*label_count).square()*mask[..., None]).sum((0, 1))
            output.register_hook(backward)
        handles.append(module.register_forward_hook(hook))
    started = time.perf_counter(); total = cfg['calibration_examples']; num_batches = math.ceil(total/cfg['calibration_batch'])
    base.zero_grad(set_to_none=True)
    for start in range(0, total, cfg['calibration_batch']):
        indices = list(range(start, min(total, start+cfg['calibration_batch'])))
        inputs = batch(data, indices); context.update(mask=inputs['attention_mask'].double(), labels=len(indices))
        # Calibration uses float32 forward/backward to avoid scaling noise in
        # small gradients; frozen model weights temporarily become float32.
        loss = forward(base, inputs, data['targets'][indices].to('cuda')).loss/num_batches
        loss.backward()
    for handle in handles:
        handle.remove()
    # Each Hout hook saw loss/num_batches. Undo that scale in the second moment.
    utility, spectra, metadata = {}, [], {}
    for index, (name, module) in enumerate(modules.items()):
        values = stats[name]; count = values['tokens']
        g = module.weight.grad.detach().T.float().cpu().numpy()
        hi = (values['Hin']/count).float().cpu().numpy()
        ho = (values['Hout']*num_batches**2/count).float().cpu().numpy()
        filename = f'module_{index:02d}.npz'
        np.savez_compressed(folder/filename, G=g, Hin=hi, Hout=ho)
        matrix = g.astype(np.float64)/np.sqrt(ho.astype(np.float64)+cfg['damping']*ho.mean(dtype=np.float64))[:, None]
        matrix /= np.sqrt(hi.astype(np.float64)+cfg['damping']*hi.mean(dtype=np.float64))[None, :]
        singular = np.linalg.svd(matrix, compute_uv=False)[:cfg['rank_cap']]
        utility[name] = np.r_[0., np.cumsum(.5*singular**2)]
        for rank, sigma in enumerate(singular, 1):
            spectra.append({'module': name, 'rank': rank, 'sigma': float(sigma), 'prefix_utility': float(utility[name][rank])})
        metadata[name] = {'file': filename, 'shape': list(g.shape), 'cost': sum(g.shape), 'index': index}
        module.weight.grad = None; module.weight.requires_grad_(False)
    write(folder/'modules.json', metadata); csv_write(folder/'spectra.csv', spectra)
    budget = len(modules)*cfg['uniform_rank']
    choices = {'UniformLoRA': {'ranks': {name: cfg['uniform_rank'] for name in modules}}}
    for name, floor in [('TwoSidedSpectral', 0), ('CoverageFloor', cfg['coverage_floor'])]:
        ranks, objective = solve_rank(modules, utility, budget, cfg['rank_cap'], floor)
        choices[name] = {'ranks': ranks, 'predicted_utility': objective}
    uniform = choices['UniformLoRA']['ranks']; spectral = choices['TwoSidedSpectral']['ranks']
    tv = sum(abs(uniform[n]-spectral[n]) for n in modules)/(2*budget)
    allocation = {'methods': choices, 'total_rank_budget': budget, 'factor_budget': budget*3072,
                  'module_parameter_tv_vs_uniform': tv, 'near_degenerate': tv < .10,
                  'calibration_seconds': time.perf_counter()-started, 'calibration_examples': total,
                  'forward_dtype': 'float32', 'SVD_dtype': 'float64_numpy_cpu'}
    write(folder/'allocations.json', allocation)
    return allocation


def attach(base, cfg, method, ranks, steps):
    if method == 'AdaLoRA':
        config = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['c_attn'],
            fan_in_fan_out=True, init_r=cfg['adalora_init_rank'], target_r=cfg['uniform_rank'],
            lora_alpha=cfg['adalora_alpha'], lora_dropout=0., total_step=steps,
            tinit=min(cfg['adalora_tinit'], max(1, steps//4)), tfinal=min(cfg['adalora_tfinal'], max(1, steps//4)),
            deltaT=1 if steps < 10 else cfg['adalora_deltaT'], orth_reg_weight=cfg['adalora_orth_reg_weight'])
    else:
        positive_targets = [name for name, rank in ranks.items() if rank > 0]
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=positive_targets,
            fan_in_fan_out=True, r=cfg['uniform_rank'], rank_pattern=ranks,
            lora_alpha=cfg['static_rslora_alpha'], use_rslora=True, lora_dropout=0., bias='none')
    adapted = get_peft_model(base, config)
    for name, parameter in adapted.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return adapted


def rank_pattern(model, cfg, method):
    result = {}
    for name, module in model.named_modules():
        if name.endswith(cfg['target_suffix']):
            key = name.removeprefix('base_model.model.')
            if not hasattr(module, 'lora_A'):
                result[key] = 0
            elif method == 'AdaLoRA':
                result[key] = int((module.lora_E['default'].detach().abs() > 0).sum())
            else:
                result[key] = int(module.r['default'])
    assert len(result) == 12
    return result


@torch.no_grad()
def numerical_proof(model, inputs):
    parameters = list(model.parameters()); buffers = list(model.buffers())
    dtypes = [(p, p.dtype) for p in parameters+buffers if p.is_floating_point()]
    model.float(); model.eval()
    full = model(**inputs, logits_to_keep=0, use_cache=False).logits[:, -1].float()
    compact = forward(model, inputs).logits[:, -1].float()
    error = float((full-compact).abs().max())
    # Compare a common arbitrary target to test the full-vocabulary CE too.
    targets = torch.zeros(len(full), dtype=torch.long, device='cuda')
    loss_error = float((F.cross_entropy(full, targets)-F.cross_entropy(compact, targets)).abs())
    merge_error = 0.
    generator = torch.Generator(device='cuda').manual_seed(19351)
    for name, module in model.named_modules():
        if name.endswith('attn.c_attn') and hasattr(module, 'get_delta_weight'):
            x = torch.randn((2, 3, 768), device='cuda', generator=generator)
            adapted = module(x)
            weight = module.base_layer.weight+module.get_delta_weight('default')
            merged = torch.addmm(module.base_layer.bias, x.reshape(-1, 768), weight).reshape_as(adapted)
            merge_error = max(merge_error, float((adapted-merged).abs().max()))
    for parameter, dtype in dtypes:
        parameter.data = parameter.data.to(dtype)
    # CE(z,y)=logsumexp(z)-z_y is 2-Lipschitz in the infinity norm.
    # GEMMs with different output shapes can reduce FP32 sums differently;
    # bind loss roundoff to the observed logits rather than an arbitrary
    # tighter CE threshold. This is fixed before any formal training.
    assert error < 5e-4 and loss_error <= 2*error+1e-6 and merge_error < 5e-4, (error, loss_error, merge_error)
    return {'full_vs_compact_max_logit_error': error, 'full_vs_compact_loss_error': loss_error,
            'fp32_module_merge_error': merge_error}


def unit_id(phase, task, method, seed, lr):
    return f'{phase}_{task}_{method}_seed{seed}_lr{lr:.0e}'


def train_unit(base, data, validation, cfg, run, phase, task, method, seed, lr, ranks, smoke):
    name = unit_id(phase, task, method, seed, lr); folder = run/'units'/name
    if (folder/'result.json').exists():
        saved_result = read(folder/'result.json')
        assert digest(folder/'adapter.pt') == saved_result['checkpoint_sha256']
        return saved_result, base
    if folder.exists() and any(folder.iterdir()):
        attempt = folder/'attempts'/str(time.time_ns())
        attempt.mkdir(parents=True)
        for previous in list(folder.iterdir()):
            if previous.is_file():
                previous.rename(attempt/previous.name)
    folder.mkdir(parents=True, exist_ok=True)
    steps = 3 if smoke else cfg['steps']
    seed_all(seed); base_refs = dict(base.named_parameters()); base_hash = parameter_hash(base_refs)
    probe_indices = [0, 1]; probe = batch(data, probe_indices)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        initial = forward(base, probe).logits.detach().clone()
    model = attach(base, cfg, method, ranks, steps)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        zero_error = float((forward(model, probe).logits-initial).abs().max())
    assert zero_error == 0., zero_error
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    physical = sum(p.numel() for p in trainable.values())
    expected = 295008 if method == 'AdaLoRA' else 147456
    assert physical == expected, (method, physical)
    optimizer = torch.optim.AdamW(trainable.values(), lr=lr, betas=tuple(cfg['adam_betas']),
                                  weight_decay=cfg['weight_decay'], fused=True)
    rng = np.random.default_rng(100000+seed)
    indices = rng.integers(0, len(data['labels']), size=(steps, cfg['accumulation_steps'], cfg['micro_batch']))
    trace = hashlib.sha256(); history = []; optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    with (folder/'training.jsonl').open('w', encoding='utf-8') as log:
        for step in range(1, steps+1):
            model.train(); ce_total, loss_total = 0., 0.
            if step <= cfg['warmup_steps']:
                factor = step/cfg['warmup_steps']
            else:
                factor = .5*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/(steps-cfg['warmup_steps'])))
            for group in optimizer.param_groups:
                group['lr'] = lr*factor
            for index in indices[step-1]:
                inputs = batch(data, index); targets = data['targets'][index].to('cuda')
                trace.update(inputs['input_ids'].cpu().numpy().astype('<i8').tobytes())
                trace.update(targets.cpu().numpy().astype('<i8').tobytes())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = forward(model, inputs, targets)
                    ce = F.cross_entropy(output.logits[:, -1].float(), targets)
                    total_loss = output.loss
                    loss = total_loss/cfg['accumulation_steps']
                assert torch.isfinite(loss), 'Nonfinite training loss'
                loss.backward(); ce_total += float(ce.detach())/cfg['accumulation_steps']
                loss_total += float(total_loss.detach())/cfg['accumulation_steps']
            torch.nn.utils.clip_grad_norm_(trainable.values(), cfg['gradient_clip'])
            optimizer.step()
            if method == 'AdaLoRA':
                model.base_model.update_and_allocate(step)
            optimizer.zero_grad(set_to_none=True)
            pattern = rank_pattern(model, cfg, method); total_rank = sum(pattern.values())
            history.append(total_rank)
            record = {'step': step, 'nll': ce_total, 'total_loss': loss_total,
                      'regularization': loss_total-ce_total, 'lr': lr*factor,
                      'effective_rank': total_rank, 'effective_factor_parameters': total_rank*3072,
                      'rank_pattern': pattern, 'data_trace_sha256': trace.hexdigest()}
            log.write(json.dumps(record)+'\n'); log.flush()
    torch.cuda.synchronize(); seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated(); final_ranks = rank_pattern(model, cfg, method)
    if not smoke:
        assert sum(final_ranks.values()) == 48, final_ranks
    metrics = evaluate(model, validation, cfg, folder/'predictions.csv')
    proof = numerical_proof(model, probe)
    final_hash = parameter_hash(base_refs)
    assert final_hash == base_hash, 'Frozen base changed'
    checkpoint = {'trainable_state': {n: p.detach().cpu().clone() for n, p in trainable.items()},
                  'method': method, 'initial_rank_pattern': ranks, 'steps': steps}
    torch.save(checkpoint, folder/'adapter.pt')
    count = sum(final_ranks.values())
    equivalent = None
    if method != 'AdaLoRA':
        choices = read(run/'calibration'/task/'allocations.json')['methods']
        for reference in ('UniformLoRA', 'TwoSidedSpectral'):
            if reference != method and choices[reference]['ranks'] == ranks:
                equivalent = reference
                break
    result = {'phase': phase, 'task': task, 'method': method, 'seed': seed, 'lr': lr,
        'steps': steps, 'smoke': smoke, 'evaluated_split': 'tune' if phase == 'tuning' else 'validation',
        'metrics': metrics, 'physical_trainable': physical, 'final_factor_parameters': count*3072,
        'final_effective_parameters': count*(3073 if method == 'AdaLoRA' else 3072),
        'final_rank_total': count, 'rank_pattern': final_ranks,
        'effective_factor_parameter_peak': max(history)*3072,
        'effective_factor_parameter_average': float(np.mean(history))*3072,
        'training_seconds': seconds, 'peak_cuda_allocated_bytes': peak, 'data_trace_sha256': trace.hexdigest(),
        'zero_init_error': zero_error, 'base_frozen': final_hash == base_hash,
        'initial_base_sha256': base_hash, 'final_base_sha256': final_hash,
        'checkpoint_sha256': digest(folder/'adapter.pt'),
        'configuration_equivalent_to': equivalent, 'independently_retrained': True, **proof}
    write(folder/'result.json', result)
    finished = [read(p) for p in (run/'units').glob('*/result.json')]
    write(run/'progress.json', {'completed_units': len(finished),
        'tuning': sum(r['phase'] == 'tuning' for r in finished),
        'main': sum(r['phase'] == 'main' for r in finished), 'last_unit': name})
    base = model.unload(); del optimizer, model, trainable, checkpoint
    base.requires_grad_(False); base.eval(); torch.cuda.empty_cache()
    print(json.dumps({'unit': name, 'metrics': metrics, 'seconds': seconds, 'physical': physical, 'rank': count}), flush=True)
    return result, base


def audit(base, datasets, cfg, run):
    output = {'records': [], 'main_checkpoints_reloaded': 0, 'complete': False, 'pass': True}
    if (run/'checkpoint_audit.json').exists():
        output = read(run/'checkpoint_audit.json')
    cached = {r['result_path']: r for r in output['records']}
    for path in sorted((run/'units').glob('main_*/result.json')):
        key = path.relative_to(run).as_posix(); result = read(path)
        if key in cached and cached[key]['checkpoint_sha256'] == result['checkpoint_sha256']:
            continue
        checkpoint_path = path.parent/'adapter.pt'; assert digest(checkpoint_path) == result['checkpoint_sha256']
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        seed_all(result['seed']); model = attach(base, cfg, result['method'], saved['initial_rank_pattern'], saved['steps'])
        outcome = model.load_state_dict(saved['trainable_state'], strict=False)
        assert not outcome.unexpected_keys
        assert set(saved['trainable_state']) == {n for n, p in model.named_parameters() if p.requires_grad}
        current = evaluate(model, datasets[result['task']]['validation'], cfg)
        errors = {k: abs(current[k]-result['metrics'][k]) for k in ('nll', 'accuracy', 'f1')}
        assert max(errors.values()) == 0., errors
        assert rank_pattern(model, cfg, result['method']) == result['rank_pattern']
        cached[key] = {'result_path': key, 'checkpoint_sha256': result['checkpoint_sha256'],
                       'max_metric_reload_error': max(errors.values()), 'pass': True}
        output.update(records=list(cached.values()), main_checkpoints_reloaded=len(cached))
        write(run/'checkpoint_audit.json', output)
        base = model.unload(); base.requires_grad_(False); del model; torch.cuda.empty_cache()
    output.update({'complete': len(cached) == 40, 'pass': True})
    write(run/'checkpoint_audit.json', output)
    print({'audit_reloads': len(cached), 'complete': output['complete']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['calibrate', 'tune', 'main', 'audit', 'all'], default='all')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); run = args.run.resolve(); cfg = read(CONFIG)
    assert torch.cuda.is_available(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    initialize(run, cfg, args.smoke)
    base, tokenizer = load_base(cfg)
    datasets = {task: {split: tokenize(cfg, tokenizer, task, split) for split in ('train', 'tune', 'validation')} for task in cfg['tasks']}
    write(run/'tokenization_audit.json', {task: {split: {'examples': len(value['labels']), 'truncated': value['truncated'],
            'candidate_ids': value['candidate_ids']} for split, value in splits.items()} for task, splits in datasets.items()})
    write(run/'model_audit.json', {'parameters': sum(p.numel() for p in base.parameters()), 'model_revision': cfg['model_revision'],
                                 'dropout_disabled': True, 'trainable_head_or_embedding': False})
    if args.phase == 'audit':
        audit(base, datasets, cfg, run); return
    if not (run/'frozen_baselines.json').exists():
        write(run/'frozen_baselines.json', {task: evaluate(base, datasets[task]['validation'], cfg) for task in cfg['tasks']})
    allocations = {}
    for task in cfg['tasks']:
        folder = run/'calibration'/task
        if not (folder/'allocations.json').exists():
            base.float()
            allocations[task] = calibrate(base, datasets[task]['train'], cfg, folder)
            base.bfloat16(); base.requires_grad_(False)
        else:
            allocations[task] = read(folder/'allocations.json')
    if args.phase == 'calibrate':
        return
    selection = read(run/'selection.json') if (run/'selection.json').exists() else {}
    if args.phase in ('all', 'tune') or args.smoke:
        for task in cfg['tasks']:
            selection.setdefault(task, {})
            for method in cfg['methods']:
                ranks = allocations[task]['methods'].get(method, allocations[task]['methods']['UniformLoRA'])['ranks']
                candidates = []
                for lr in (cfg['learning_rates'][:1] if args.smoke else cfg['learning_rates']):
                    result, base = train_unit(base, datasets[task]['train'], datasets[task]['tune'], cfg, run,
                        'tuning', task, method, cfg['tuning_seed'], lr, ranks, args.smoke)
                    candidates.append({'lr': lr, 'tuning_nll': result['metrics']['nll'],
                                       'unit_id': unit_id('tuning', task, method, cfg['tuning_seed'], lr)})
                best = min(candidates, key=lambda row: (row['tuning_nll'], row['lr']))
                selection[task][method] = {**best, 'candidates': candidates}
                write(run/'selection.json', selection)
        if args.phase == 'tune' and not args.smoke:
            return
    if args.smoke:
        write(run/'completion.json', {'complete': True, 'smoke': True, 'tuning': 8, 'main': 0})
        return
    assert all(len(selection.get(task, {})) == 4 for task in cfg['tasks'])
    selection_lock = {'selection_sha256': digest(run/'selection.json'),
                      'tuning_units': 24, 'official_validation_used_for_selection': False}
    if (run/'selection_lock.json').exists():
        assert read(run/'selection_lock.json') == selection_lock
    else:
        write(run/'selection_lock.json', selection_lock)
    for task in cfg['tasks']:
        for seed in cfg['main_seeds']:
            for method in cfg['methods']:
                ranks = allocations[task]['methods'].get(method, allocations[task]['methods']['UniformLoRA'])['ranks']
                _, base = train_unit(base, datasets[task]['train'], datasets[task]['validation'], cfg, run,
                    'main', task, method, seed, selection[task][method]['lr'], ranks, False)
    audit(base, datasets, cfg, run)
    for item in read(run/'prior_artifacts.json'):
        assert digest(ROOT/item['path']) == item['sha256'], item['path']
    units = [read(path) for path in (run/'units').glob('*/result.json')]
    assert len(units) == 64
    write(run/'completion.json', {'complete': True, 'smoke': False, 'tuning': 24, 'main': 40,
        'actual_optimizer_trajectories': 64, 'old_artifacts_unchanged': True, 'test_used': False})
    print('Real-model protocol completed', flush=True)


if __name__ == '__main__':
    main()
