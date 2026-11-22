"""Bounded true-versus-five-random-group experiment after the capacity gate."""
from __future__ import annotations

import argparse
import copy
import shutil

import numpy as np
import torch

from .allocation import allocate
from .capacity import dataset
from .curvature import spectral_utility, whiten
from .diagnostic import Diagnostic
from .experiment import ROOT, Experiment, base_digest, digest, read_json, write_csv, write_json
from .model import ModelConfig, structure
from scripts.calibrate_output_diagonal import collect


def group_schemes(widths, shuffle_seeds):
    rec = structure(widths)[-1]
    schemes = {}
    for scheme in ['true'] + [f'shuffle_{s}' for s in shuffle_seeds]:
        groups = []
        rng = np.random.default_rng(int(scheme.split('_')[1])) if scheme != 'true' else None
        for projection in ('q_proj', 'v_proj'):
            name = f'blocks.{len(widths)-1}.attn.{projection}'
            permutation = rng.permutation(rec['width']).tolist() if rng is not None else None
            for index, label in enumerate(('C', 'R')):
                indices = rec[label] if rng is None else sorted(permutation[64*index:64*(index+1)])
                groups.append({'id': name + ':' + label, 'module': name, 'label': label,
                               'indices': indices, 'n': len(indices), 'm': rec['width'],
                               'cost': len(indices) + rec['width']})
        assert all(g['n'] == 64 and g['cost'] == 192 for g in groups)
        schemes[scheme] = groups
    # Distinct unordered partitions: a label swap must not count as a new grouping.
    for module in {g['module'] for g in schemes['true']}:
        signatures = [tuple(sorted(tuple(g['indices']) for g in groups if g['module'] == module))
                      for groups in schemes.values()]
        assert len(signatures) == len(set(signatures))
    return schemes


class LabelExperiment(Diagnostic):
    def __init__(self, output, config, resume=False):
        self.output, self.cfg, self.smoke = output, config, config['smoke']
        self.parent = ROOT / config['capacity_parent']
        if not read_json(self.parent / 'capacity_gate.json')['pass']:
            raise RuntimeError('Capacity gate failed; label experiment must not run')
        self.parent_cfg = read_json(ROOT / 'configs/pilot.json')
        self.splits, data = dataset(self.parent_cfg)
        self.parent_cfg['evaluation']['max_blocks'] = 512
        self.resume = resume
        source_paths = list((ROOT / 'src').rglob('*.py')) + [ROOT / 'scripts/calibrate_output_diagonal.py', ROOT / 'scripts/prepare_wikitext.py']
        source = {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in sorted(source_paths)}
        identity = {'config': config, 'parent_training_config': self.parent_cfg, 'sources': source, 'data': data,
                    'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0),
                    'protocol_sha256': digest((ROOT / 'plan/crossdomain-label-protocol.md').read_bytes()),
                    'inherited_base_file_sha256': {p: digest((self.parent / f'{p}_base_2026/base.pt').read_bytes()) for p in config['widths']}}
        if output.exists():
            if not resume or read_json(output / 'identity.json') != identity:
                raise ValueError('Resume requires matching identity')
        else:
            if resume:
                raise FileNotFoundError(output)
            output.mkdir(parents=True)
            write_json(output / 'identity.json', identity)
            write_json(output / 'config.json', config)
            write_json(output / 'data_manifest.json', data)
            shutil.copy2(ROOT / 'plan/crossdomain-label-protocol.md', output / 'protocol.md')
            for name in source:
                destination = output / 'source_snapshot' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
        self.rows = []

    def prepare(self, profile, seed):
        child = self.output / f'{profile}_base_{seed}'
        child.mkdir(exist_ok=True)
        cfg = copy.deepcopy(self.parent_cfg)
        cfg['name'] = f'crossdomain_labels_{profile}_base_{seed}'
        cfg['model']['widths'] = self.cfg['widths'][profile]
        cfg['pretrain']['seed'] = seed
        cfg['runtime']['log_every'] = 120
        if self.smoke:
            cfg['evaluation']['max_blocks'] = 4
        write_json(child / 'config.json', cfg)
        write_json(child / 'structure.json', structure(cfg['model']['widths']))
        if seed == 2026 and not (child / 'base.pt').exists():
            for name in ('base.pt', 'pretrain.json'):
                shutil.copy2(self.parent / f'{profile}_base_2026' / name, child / name)
            write_json(child / 'parent_checkpoint.json', {'explicit_inheritance': True,
                'source': f'{self.cfg["capacity_parent"]}/{profile}_base_2026',
                'file_sha256': digest((child / 'base.pt').read_bytes())})
        experiment = object.__new__(Experiment)
        experiment.config, experiment.output, experiment.splits = cfg, child, self.splits
        experiment.device, experiment.dtype = torch.device('cuda'), torch.bfloat16
        experiment.model_config = ModelConfig(**cfg['model'])
        experiment.pretrain()
        model, base = experiment.load_base()
        if not base['summary']['pretraining_gate']:
            raise ValueError('Base quality gate failed')
        write_json(child / 'baseline.json', {s: experiment.evaluate(model, s) for s in ('source_val', 'target_val')})
        schemes = group_schemes(cfg['model']['widths'], self.cfg['shuffle_seeds'])
        write_json(child / 'groups.json', schemes)
        destination = child / 'allocations.json'
        if destination.exists() and self.resume:
            allocations = read_json(destination)
        else:
            stats, _, _, _, _ = collect(model, self.splits, cfg)
            assert base_digest(model) == base['sha256']
            matrices, metadata = {}, {}
            for number, name in enumerate(sorted({g['module'] for g in schemes['true']})):
                metadata[name] = {}
                for key in ('G', 'Hin', 'Hout'):
                    field = f'm{number}_{key}'
                    matrices[field] = stats[name][key].numpy()
                    metadata[name][key] = field
            np.savez_compressed(child / 'calibration_arrays.npz', **matrices)
            write_json(child / 'calibration_arrays.json', metadata)
            allocations, spectra = {}, []
            for scheme, groups in schemes.items():
                utilities = {}
                for group in groups:
                    value = stats[group['module']]
                    matrix = whiten(value['G'][:, group['indices']], value['Hin'][group['indices']], value['Hout'],
                                    input_damping=self.cfg['damping_in'], output_damping=self.cfg['damping_out'],
                                    input_reference_mean=value['Hin'].mean())
                    sigma, utility = spectral_utility(matrix, self.cfg['max_rank'])
                    utilities[group['id']] = utility
                    for rank, singular in enumerate(sigma.tolist(), 1):
                        spectra.append({'scheme': scheme, 'group': group['id'], 'rank': rank,
                                        'singular_value': singular, 'prefix_utility': utility[rank]})
                ranks, spent, score = allocate(groups, utilities, self.cfg['budget'], self.cfg['max_rank'])
                assert spent == self.cfg['budget']
                allocations[scheme] = {'scheme': scheme, 'ranks': ranks, 'spent': spent, 'predicted_utility': score}
            ranks = {g['id']: int(g['module'].endswith('v_proj')) for g in schemes['true']}
            assert sum(g['cost'] * ranks[g['id']] for g in schemes['true']) == self.cfg['budget']
            allocations['VBalanced'] = {'scheme': 'true', 'ranks': ranks, 'spent': self.cfg['budget']}
            write_json(destination, allocations)
            write_csv(child / 'spectra.csv', spectra)
        del model
        return experiment, schemes, allocations

    def run(self):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        for profile in self.cfg['widths']:
            for seed in self.cfg['base_seeds']:
                experiment, schemes, allocations = self.prepare(profile, seed)
                for adapter_seed in self.cfg['adapter_seeds']:
                    for method, assignment in allocations.items():
                        if self.smoke and method not in ('true', 'shuffle_11', 'VBalanced'):
                            continue
                        rows = self.train(experiment, seed, adapter_seed, 'labels', method, 'rs_ref3',
                                          self.cfg['lr'], schemes[assignment['scheme']], assignment['ranks'], self.cfg['budget'])
                        self.rows.extend({'profile': profile, **row} for row in rows)
                        write_csv(self.output / 'main.csv', self.rows)
        expected = len(self.cfg['widths']) * len(self.cfg['base_seeds']) * len(self.cfg['adapter_seeds']) * (3 if self.smoke else 7)
        assert len(self.rows) == expected * len(self.cfg['eval_steps'])
        write_json(self.output / 'completion.json', {'trajectories': expected, 'validation_rows': len(self.rows),
                   'smoke': self.smoke, 'test_split_used': False, 'complete': True})
        print({'stage': 'label_screen_finished', 'trajectories': expected}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default=str(ROOT / 'configs/crossdomain_labels.json'))
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output).resolve()
    if output.parent != ROOT / 'runs':
        raise ValueError('Choose a direct runs/<run-id> directory')
    cfg = read_json(args.config)
    if args.smoke:
        cfg.update(smoke=True, base_seeds=[2026], adapter_seeds=[11], steps=3, eval_steps=[3])
    LabelExperiment(output, cfg, args.resume).run()


if __name__ == '__main__':
    main()
