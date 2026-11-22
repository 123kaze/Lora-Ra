"""Phase-2 controlled diagnostics; phase-1 implementation and runs stay intact."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .allocation import allocate, equal_parameters
from .data import digest, random_batch
from .experiment import (ROOT, Experiment, base_digest, code_manifest, log, new_attempt,
                         read_json, seed_all, synchronize, write_csv, write_json)
from .lora import GroupedLinear, attach, merge_all


def rank_scale(mode, rank):
    if mode not in ('constant','rs_ref3') or rank <= 0:
        raise ValueError((mode,rank))
    return 1.0 if mode == 'constant' else math.sqrt(3.0 / rank)


def input_utility(gradient,moment,max_rank,damping=0.01):
    diagonal=moment+damping*moment.mean()
    if (diagonal<=0).any():
        raise ValueError('Nonpositive input second moment')
    singular=torch.linalg.svdvals(gradient/diagonal.sqrt())[:max_rank]
    return [0.0]+(0.5*singular.square()).cumsum(0).tolist(),singular.tolist()


def reserve_q(groups, utilities, budget, cap):
    """Exact rank-prefix optimization with at least one Q rank per projection."""
    q_modules = sorted({g['module'] for g in groups if g['module'].endswith('q_proj')})
    choices = [[g['id'] for g in groups if g['module'] == module] for module in q_modules]
    best = None
    for selected in itertools.product(*choices):
        floor = {gid: 1 for gid in selected}
        minimum_cost = sum(g['cost'] * floor.get(g['id'], 0) for g in groups)
        if minimum_cost > budget:
            continue
        shifted, values = [], {}
        constant = 0.0
        for g in groups:
            gid, low = g['id'], floor.get(g['id'], 0)
            limit = min(cap, g['m'], g['n']) - low
            shifted.append({**g, 'm': limit})
            initial = utilities[gid][low]
            constant += initial
            values[gid] = [v - initial for v in utilities[gid][low:low + limit + 1]]
        try:
            extra, spent, score = allocate(shifted, values, budget - minimum_cost, cap)
        except ValueError:
            continue
        ranks = {g['id']: extra[g['id']] + floor.get(g['id'], 0) for g in groups}
        if best is None or score + constant > best[2]:
            best = ranks, spent + minimum_cost, score + constant
    if best is None:
        raise ValueError('Q reserve is not feasible at this exact budget')
    return best


def input_moments(experiment, model):
    sums, counts, handles = {}, {}, []
    for name, linear, _ in model.targets():
        sums[name], counts[name] = torch.zeros(linear.in_features, dtype=torch.float64), 0
        def collect(module, inputs, key=name):
            x = inputs[0].detach().reshape(-1, module.in_features)
            sums[key].add_(x.double().square().sum(0).cpu())
            counts[key] += x.shape[0]
        handles.append(linear.register_forward_pre_hook(collect))
    cfg = experiment.config['calibration']
    rng = torch.Generator().manual_seed(cfg['seed'])
    model.eval()
    with torch.no_grad():
        for _ in range(cfg['batches']):
            x, _ = random_batch(experiment.splits['target_train'], cfg['batch_size'],
                                experiment.model_config.seq_len, rng, experiment.device)
            model(x)
    for handle in handles:
        handle.remove()
    return {name: value / counts[name] for name, value in sums.items()}


class Diagnostic:
    def __init__(self, output, config, resume=False, smoke=False):
        self.output, self.cfg = output, copy.deepcopy(config)
        self.smoke = smoke
        if smoke:
            self.cfg.update(base_seeds=[2026], adapter_seeds=[11], steps=3, eval_steps=[3],
                            scales=['constant'], sweep_ranks=[1], concentration_methods=['Gradient'],
                            concentration_lrs=[0.003])
        self.parent = ROOT / self.cfg['parent_run']
        self.parent_cfg = read_json(self.parent / 'config.json')
        # Check all recorded parent files; additions are new diagnostic code, not altered old code.
        for relative, expected in read_json(self.parent / 'identity.json')['code'].items():
            if digest((ROOT / relative).read_bytes()) != expected:
                raise ValueError(f'Parent source changed: {relative}')
        identity = {'config': self.cfg, 'code': code_manifest(), 'smoke': smoke,
                    'parent_base_file_sha256': digest((self.parent / 'base.pt').read_bytes())}
        if output.exists():
            if not resume or read_json(output / 'diagnostic_identity.json') != identity:
                raise ValueError('Existing diagnostic run requires matching identity and --resume')
        else:
            output.mkdir(parents=True)
            write_json(output / 'diagnostic_identity.json', identity)
            write_json(output / 'diagnostic_config.json', self.cfg)
            shutil.copy2(ROOT / 'plan/diagnostic-protocol.md', output / 'protocol.md')
        self.rows = []

    def prepare_base(self, seed):
        config = copy.deepcopy(self.parent_cfg)
        config['name'] = f'diagnostic_base_{seed}'
        config['pretrain']['seed'] = seed
        child = self.output / f'base_{seed}'
        experiment = Experiment(config, child, resume=child.exists())
        if seed == 2026 and not (child / 'base.pt').exists():
            shutil.copy2(self.parent / 'base.pt', child / 'base.pt')
            shutil.copy2(self.parent / 'pretrain.json', child / 'pretrain.json')
            write_json(child / 'parent_checkpoint.json', {'source':self.cfg['parent_run'],
                       'file_sha256':digest((child / 'base.pt').read_bytes()), 'explicit_inheritance':True})
        experiment.pretrain()
        calibration = experiment.calibrate()
        model, base = experiment.load_base()
        if not base['summary']['pretraining_gate']:
            raise RuntimeError('Pretraining quality gate failed')
        moments = input_moments(experiment, model)
        assert base_digest(model) == base['sha256']
        normalized, records = {}, []
        for scheme, groups in calibration['groups'].items():
            normalized[scheme] = {}
            for g in groups:
                moment = moments[g['module']]
                matrix = calibration['gradients'][g['module']][:, g['indices']]
                z = matrix / (moment[g['indices']] + self.cfg['input_damping'] * moment.mean()).sqrt()
                sigma = torch.linalg.svdvals(z)[:self.parent_cfg['adapter']['max_rank']]
                normalized[scheme][g['id']] = [0.0] + (0.5 * sigma.square()).cumsum(0).tolist()
                for j, value in enumerate(sigma.tolist(), 1):
                    records.append({'base_seed':seed,'scheme':scheme,'group':g['id'],'rank':j,
                                    'label':g['label'],'n':g['n'],'m':g['m'],
                                    'input_second_moment':moment[g['indices']].mean().item(),
                                    'raw_utility':calibration['utilities'][scheme][g['id']][j],
                                    'diag_utility':normalized[scheme][g['id']][j]})
        write_csv(child / 'normalized_spectra.csv', records)
        write_json(child / 'input_second_moments.json', {k:v.tolist() for k,v in moments.items()})
        write_json(child / 'baseline_validation.json', experiment.evaluate(model, 'target_val'))
        del model
        return experiment, calibration, normalized

    def allocation(self, method, calibration, normalized, budget):
        cap = self.parent_cfg['adapter']['max_rank']
        scheme = 'shuffled' if 'Shuffled' in method else 'true'
        groups = calibration['groups'][scheme]
        if method == 'VOnlyBalanced':
            groups = [g for g in groups if g['module'].endswith('v_proj')]
        if method in ('Balanced', 'VOnlyBalanced'):
            ranks, spent, _ = equal_parameters(groups, budget, cap)
        else:
            utilities = normalized[scheme] if method.startswith('Diag') else calibration['utilities'][scheme]
            solver = reserve_q if method == 'GradientQFloor' else allocate
            ranks, spent, _ = solver(groups, utilities, budget, cap)
        assert spent == budget
        return groups, ranks

    def train(self, experiment, base_seed, adapter_seed, regime, method, mode, lr, groups, ranks, budget):
        key = f'{regime}_{method}_{mode}_lr{lr:g}_seed{adapter_seed}'
        unit = experiment.output / 'diagnostic_adapters' / key
        if (unit / 'result.json').exists():
            return read_json(unit / 'result.json')['evaluations']
        unit.mkdir(parents=True, exist_ok=True)
        seed_all(adapter_seed)
        model, base = experiment.load_base()
        probe = experiment.splits['target_train'][:experiment.model_config.seq_len][None].to(experiment.device)
        model.eval()
        with torch.no_grad():
            original = model(probe)
        seed_all(adapter_seed)
        params = attach(model, groups, ranks, 1.0)
        for module in model.modules():
            if isinstance(module, GroupedLinear):
                for factor in module.factors:
                    factor.scaling = rank_scale(mode, factor.A.shape[0])
        with torch.no_grad():
            init_error = (model(probe)-original).abs().max().item()
        assert init_error == 0 and params == budget
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        rng = torch.Generator().manual_seed(adapter_seed + 100000)
        trace = hashlib.sha256()
        total, since = 0.0, 0
        evaluations, updates = [], []
        started = time.perf_counter()
        with new_attempt(experiment.output, key) as handle:
            for step in range(1, self.cfg['steps']+1):
                model.train()
                x,y = random_batch(experiment.splits['target_train'], self.parent_cfg['adapter']['batch_size'],
                                   experiment.model_config.seq_len,rng,experiment.device)
                trace.update(x.cpu().numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                with experiment.autocast():
                    loss = F.cross_entropy(model(x).flatten(0,1),y.flatten())
                if not torch.isfinite(loss):
                    raise FloatingPointError(key)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                optimizer.step()
                total += loss.item()
                since += 1
                if step in (1,60,180,600) or step == self.cfg['steps']:
                    delta_squared = 0.0
                    for name,module in model.named_modules():
                        if isinstance(module,GroupedLinear):
                            for index,factor in enumerate(module.factors):
                                delta = (factor.B.detach()@factor.A.detach())*factor.scaling
                                norm = delta.norm().item()
                                delta_squared += norm**2
                                updates.append({'base_seed':base_seed,'adapter_seed':adapter_seed,'regime':regime,
                                    'method':method,'mode':mode,'lr':lr,'step':step,'module':name,'factor':index,
                                    'rank':factor.A.shape[0],'scaling':factor.scaling,'delta_frobenius':norm,
                                    'relative_to_base_projection':norm/module.base.weight.detach().norm().item()})
                    log(handle,{'stage':'diagnostic_train','method':method,'regime':regime,'mode':mode,
                                'base_seed':base_seed,'adapter_seed':adapter_seed,'lr':lr,'step':step,
                                'interval_train_loss':total/since,'delta_frobenius':math.sqrt(delta_squared)})
                    total,since = 0.0,0
                if step in self.cfg['eval_steps']:
                    metric = experiment.evaluate(model,'target_val')
                    evaluations.append({'base_seed':base_seed,'adapter_seed':adapter_seed,'regime':regime,
                        'method':method,'mode':mode,'lr':lr,'budget':budget,'step':step,'val_loss':metric['loss'],
                        'val_accuracy':metric['accuracy'],'val_bytes':metric['bytes'],
                        'base_val_loss':base['summary']['target_val']['loss'],
                        'improvement':base['summary']['target_val']['loss']-metric['loss'],
                        'data_trace_sha256':trace.hexdigest(),'base_sha256':base['sha256']})
                    log(handle,{'stage':'diagnostic_validation',**evaluations[-1]})
            synchronize(experiment.device)
            frozen = base_digest(model) == base['sha256']
            assert frozen
            model.eval()
            with torch.no_grad():
                before_merge = model(probe)
            torch.save({'adapter':{k:v.detach().cpu() for k,v in model.state_dict().items() if '.factors.' in k},
                        'groups':groups,'ranks':ranks,'mode':mode,'base_sha256':base['sha256']},unit/'adapter.pt')
            merge_all(model)
            with torch.no_grad():
                error = (model(probe)-before_merge).abs().max().item()
            assert error < 1e-4
            result = {'evaluations':evaluations,'groups':groups,'ranks':ranks,'zero_initial_error':init_error,
                      'base_frozen':frozen,'merge_max_error':error,'time_s':time.perf_counter()-started}
            write_csv(unit/'updates.csv',updates)
            write_json(unit/'result.json',result)
            log(handle,{'stage':'diagnostic_complete','key':key,'merge_max_error':error,'base_frozen':frozen})
        del model,optimizer
        torch.cuda.empty_cache()
        return evaluations

    def add(self, *args):
        self.rows.extend(self.train(*args))
        write_csv(self.output/'diagnostic.csv',self.rows)

    def run(self):
        for base_seed in self.cfg['base_seeds']:
            experiment,calibration,normalized = self.prepare_base(base_seed)
            budget = self.cfg['budgets']['small']
            for adapter_seed in self.cfg['adapter_seeds']:
                for mode in self.cfg['scales']:
                    for method in self.cfg['main_methods']:
                        groups,ranks = self.allocation(method,calibration,normalized,budget)
                        self.add(experiment,base_seed,adapter_seed,'small',method,mode,0.003,groups,ranks,budget)
            if base_seed != 2026:
                continue
            for mode in self.cfg['scales']:
                for group in calibration['groups']['true']:
                    if group['module'] != 'blocks.4.attn.v_proj':
                        continue
                    for rank in self.cfg['sweep_ranks']:
                        self.add(experiment,base_seed,11,'capacity',f"{group['label']}_rank{rank}",mode,0.003,
                                 [group],{group['id']:rank},rank*group['cost'])
            for method in self.cfg['concentration_methods']:
                budget = self.cfg['budgets']['original']
                groups,ranks = self.allocation(method,calibration,normalized,budget)
                for mode,lr in [('constant',0.003),('rs_ref3',0.003),('constant',0.001)]:
                    if self.smoke and (mode != 'constant' or lr != 0.003):
                        continue
                    self.add(experiment,base_seed,11,'concentration',method,mode,lr,groups,ranks,budget)
        from .diagnostic_report import build
        build(self.output,self.cfg,self.smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/diagnostic.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT/'runs':
        parser.error('Choose a fresh direct runs/<run-id> directory')
    Diagnostic(output,read_json(args.config),args.resume,args.smoke).run()


if __name__ == '__main__':
    main()
