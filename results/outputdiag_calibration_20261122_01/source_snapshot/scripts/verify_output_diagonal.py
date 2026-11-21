"""Recompute all calibration spectra and exact rank allocations with NumPy only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def solve(groups,utilities,budget,cap):
    states={0:(0.0,{})}
    for group in groups:
        next_states={}
        for spent,(value,ranks) in states.items():
            for rank in range(min(cap,group['m'],group['n'])+1):
                total=spent+rank*group['cost']
                if total>budget:break
                score=value+utilities[group['id']][rank]
                if total not in next_states or score>next_states[total][0]:
                    next_states[total]=(score,{**ranks,group['id']:rank})
        states=next_states
    return states[budget][1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    args=parser.parse_args();bundle=args.bundle.resolve()
    manifest=read(bundle/'artifact-manifest.json')
    files=list(manifest['files'])
    if (bundle/'supplement-manifest.json').exists():
        files.extend(read(bundle/'supplement-manifest.json')['files'])
    for item in files:
        path=(ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT):raise ValueError(item['path'])
        content=path.read_bytes()
        if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:
            raise AssertionError('Artifact hash mismatch: '+item['path'])
    import csv
    with (bundle/'spectra.csv').open(newline='') as handle:records=list(csv.DictReader(handle))
    expected={(int(r['base_seed']),float(r['q_multiplier']),r['scheme'],r['metric'],r['group'],int(r['rank'])):float(r['prefix_utility']) for r in records}
    allocations=read(bundle/'allocations.json')
    config=read(bundle/'config.json')
    cache={};max_error=0.;count=0;allocation_checks=0
    for seed in config['base_seeds']:
        child=bundle/f'base_{seed}'
        groups=read(child/'groups.json')
        baseline=np.load(child/'scale_1p0.npz',allow_pickle=False)
        for scale in config['scales']:
            token=str(scale).replace('.','p')
            metadata=read(child/f'scale_{token}.json')['modules']
            arrays=np.load(child/f'scale_{token}.npz',allow_pickle=False)
            for scheme in config['schemes']:
                for metric in config['metrics']+['fixed_absolute_out_control']:
                    utilities={}
                    for group in groups[scheme]:
                        keys=metadata[group['module']]
                        gradient=arrays[keys['G']][:,group['indices']]
                        hi=arrays[keys['Hin']];ho=arrays[keys['Hout']]
                        input_diagonal=(hi+config['damping_in']*hi.mean())[group['indices']]
                        if metric=='raw':matrix=gradient
                        elif metric=='diag_input':matrix=gradient/np.sqrt(input_diagonal)
                        else:
                            epsilon=config['damping_out']*(ho.mean() if metric=='diag_both' else baseline[keys['Hout']].mean())
                            matrix=gradient/np.sqrt(ho+epsilon)[:,None]/np.sqrt(input_diagonal)
                        singular=np.linalg.svd(matrix,compute_uv=False)[:config['max_rank']]
                        utility=np.r_[0.,np.cumsum(.5*singular**2)]
                        utilities[group['id']]=utility
                        for rank in range(1,len(utility)):
                            max_error=max(max_error,abs(utility[rank]-expected[(seed,scale,scheme,metric,group['id'],rank)]))
                            count+=1
                    cache[(seed,scale,scheme,metric)]=utilities
            arrays.close()
        baseline.close()
    for item in allocations:
        seed,scale,scheme,metric=(item[k] for k in ['base_seed','q_multiplier','scheme','metric'])
        groups=read(bundle/f'base_{seed}/groups.json')[scheme]
        ranks=solve(groups,cache[(seed,scale,scheme,metric)],item['budget'],config['max_rank'])
        if ranks!=item['ranks']:raise AssertionError(('Allocation differs',item,ranks))
        allocation_checks+=1
    if max_error>1e-10:raise AssertionError(max_error)
    print(json.dumps({'files_verified':len(files),'rank_prefix_utilities_recomputed':count,
        'max_numpy_utility_absolute_error':max_error,'allocations_recomputed_and_identical':allocation_checks,
        'requires_torch':False,'training_steps':0,'pass':True},indent=2))


if __name__=='__main__':main()
