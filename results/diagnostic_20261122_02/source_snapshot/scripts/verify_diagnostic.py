"""Verify phase-2 public evidence without torch or the original runs directory.

--spectra additionally requires NumPy and recomputes calibration spectra from NPZ.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def rows(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--spectra',action='store_true')
    args=parser.parse_args()
    bundle=args.bundle.resolve()
    manifests=[bundle/'artifact-manifest.json',bundle/'supplement-manifest.json']
    count=0
    for manifest in manifests:
        for item in read(manifest)['files']:
            path=(ROOT/item['path']).resolve()
            if not path.is_relative_to(ROOT):raise ValueError(item['path'])
            content=path.read_bytes()
            if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:
                raise AssertionError('Artifact hash mismatch: '+str(path))
            count+=1
    data=rows(bundle/'diagnostic.csv')
    config=read(bundle/'diagnostic_config.json')
    trajectories=(len(config['base_seeds'])*len(config['adapter_seeds'])*len(config['scales'])*len(config['main_methods'])
                  +2*len(config['scales'])*len(config['sweep_ranks'])+3*len(config['concentration_methods']))
    expected=trajectories*len(config['eval_steps'])
    if len(data)!=expected:raise AssertionError('Unexpected evaluation count')
    ids={(r['base_seed'],r['adapter_seed'],r['regime'],r['method'],r['mode'],r['lr'],r['step']) for r in data}
    if len(ids)!=len(data):raise AssertionError('Duplicate evaluations')
    for summary in rows(bundle/'by_base_summary.csv'):
        selected=[r for r in data if r['regime']=='small' and all(r[k]==summary[k] for k in ('base_seed','method','mode','step'))]
        if len(selected)!=len(config['adapter_seeds']):raise AssertionError('Missing adapter seed')
        mean=statistics.mean(float(r['val_loss']) for r in selected)
        if abs(mean-float(summary['val_loss_mean']))>1e-12:raise AssertionError('Summary mismatch')
    max_spectral_error=None
    if args.spectra:
        import numpy as np
        max_spectral_error=0.0
        for base in config['base_seeds']:
            child=bundle/f'base_{base}'
            arrays=np.load(child/'calibration_arrays.npz',allow_pickle=False)
            mapping=read(child/'calibration_arrays.json')['modules']
            groups={}
            for scheme,values in read(child/'calibration.json')['groups'].items():
                for g in values:groups[(scheme,g['id'])]=g
            caches={}
            for row in rows(child/'normalized_spectra.csv'):
                key=(row['scheme'],row['group'])
                if key not in caches:
                    g=groups[key];keys=mapping[g['module']]
                    gradient=arrays[keys['gradient']][:,g['indices']]
                    moment=arrays[keys['input_second_moment']]
                    normalized=gradient/np.sqrt(moment[g['indices']]+config['input_damping']*moment.mean())
                    caches[key]=(.5*np.linalg.svd(gradient,compute_uv=False)**2,.5*np.linalg.svd(normalized,compute_uv=False)**2)
                rank=int(row['rank'])
                for index,column in [(0,'raw_utility'),(1,'diag_utility')]:
                    error=abs(float(caches[key][index][:rank].sum())-float(row[column]))
                    max_spectral_error=max(error,max_spectral_error)
        if max_spectral_error>1e-10:raise AssertionError(max_spectral_error)
    print(json.dumps({'files_verified':count,'evaluation_rows':len(data),'trajectory_count':len(data)//len(config['eval_steps']),
        'summaries_recomputed':True,'numpy_spectral_max_error':max_spectral_error,
        'requires_torch':False,'requires_original_runs_directory':False,'pass':True},indent=2))


if __name__=='__main__':main()
