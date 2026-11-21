"""Publish calibration matrices in NumPy format so audit does not require torch."""
import argparse
import numpy as np
import torch
from carryrank.experiment import ROOT, read_json, write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run')
args=parser.parse_args()
run=ROOT/args.run
published=ROOT/'results'/run.name
for child in sorted(run.glob('base_*')):
    calibration=torch.load(child/'calibration.pt',map_location='cpu',weights_only=True)
    moments=read_json(child/'input_second_moments.json')
    arrays,mapping={},{}
    for name,gradient in calibration['gradients'].items():
        key=name.replace('.','_')
        arrays['gradient__'+key]=gradient.numpy()
        arrays['moment__'+key]=np.asarray(moments[name],dtype=np.float64)
        mapping[name]={'gradient':'gradient__'+key,'input_second_moment':'moment__'+key}
    destination=published/child.name/'calibration_arrays.npz'
    if destination.exists(): raise FileExistsError(destination)
    np.savez_compressed(destination,**arrays)
    write_json(destination.with_suffix('.json'),{'format':'numpy savez_compressed; allow_pickle=False',
        'modules':mapping,'base_sha256':calibration['base_sha256'],
        'gradient':'mean task loss gradient from training calibration only',
        'input_second_moment':'uncentered E[x^2] of actual projection input',
        'dtype':'float64','no_model_weights':True})
print('Exported portable calibration for',len(list(run.glob('base_*'))),'base models')
