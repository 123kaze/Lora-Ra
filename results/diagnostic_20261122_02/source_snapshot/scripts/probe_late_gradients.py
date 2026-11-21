"""Post-hoc read-only gradient probe at final adapters; no rank reallocation/training."""
import argparse
import json

import torch
from torch.nn import functional as F

from carryrank.data import random_batch
from carryrank.diagnostic import rank_scale
from carryrank.experiment import ROOT, Experiment, base_digest, read_json, write_csv, write_json
from carryrank.lora import GroupedLinear, attach


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args=parser.parse_args()
    root=(ROOT/args.run).resolve()
    config=read_json(root/'diagnostic_config.json')
    records=[]
    for child in sorted(root.glob('base_*')):
        experiment=Experiment(read_json(child/'config.json'),child,resume=True)
        calibration=torch.load(child/'calibration.pt',map_location='cpu',weights_only=True)
        for method in ('Gradient','GradientQFloor'):
            for mode in ('constant','rs_ref3'):
                unit=child/'diagnostic_adapters'/f'small_{method}_{mode}_lr0.003_seed11'
                result=read_json(unit/'result.json')
                saved=torch.load(unit/'adapter.pt',map_location='cpu',weights_only=True)
                model,base=experiment.load_base()
                attach(model,saved['groups'],saved['ranks'],1.0)
                for module in model.modules():
                    if isinstance(module,GroupedLinear):
                        for factor in module.factors:
                            factor.scaling=rank_scale(mode,factor.A.shape[0])
                state=model.state_dict();state.update(saved['adapter']);model.load_state_dict(state)
                original_hash=base_digest(model)
                assert original_hash==base['sha256']
                model.requires_grad_(False)
                targets=[]
                for name,module,_ in model.targets():
                    linear=module.base if isinstance(module,GroupedLinear) else module
                    linear.weight.requires_grad_(True)
                    targets.append((name,linear))
                gradients={name:torch.zeros_like(linear.weight,device='cpu',dtype=torch.float64) for name,linear in targets}
                rng=torch.Generator().manual_seed(experiment.config['calibration']['seed'])
                batches=experiment.config['calibration']['batches']
                model.eval()
                for _ in range(batches):
                    x,y=random_batch(experiment.splits['target_train'],experiment.config['calibration']['batch_size'],
                                     experiment.model_config.seq_len,rng,experiment.device)
                    model.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
                    loss.backward()
                    for name,linear in targets:
                        gradients[name].add_(linear.weight.grad.detach().cpu().double()/batches)
                assert base_digest(model)==original_hash
                for group in calibration['groups']['true']:
                    for stage,gradient in [('initial',calibration['gradients'][group['module']]),('after600',gradients[group['module']])]:
                        matrix=gradient[:,group['indices']]
                        spectrum=torch.linalg.svdvals(matrix)
                        records.append({'base_seed':experiment.config['pretrain']['seed'],'adapter_seed':11,
                            'method':method,'mode':mode,'stage':stage,'group':group['id'],
                            'projection':group['module'].split('.')[-1],'label':group['label'],
                            'gradient_frobenius':matrix.norm().item(),'first_rank_utility_per_parameter':
                            (0.5*spectrum[0].square()/group['cost']).item()})
                del model
    destination=ROOT/'results'/root.name
    if (destination/'late_gradient_probe.csv').exists():
        raise FileExistsError('Late gradient probe already exists; refusing overwrite')
    write_csv(destination/'late_gradient_probe.csv',records)
    write_json(destination/'late_gradient_probe_provenance.json',{
        'purpose':'Post-hoc temporal gradient diagnostic, not a confirmatory test',
        'training_performed':False,'rank_reallocated':False,'data':'same target training calibration only',
        'base_weights_unchanged':True,'adapter_seed':11,'rows':len(records)})
    print(json.dumps({'rows':len(records),'no_training':True,'base_unchanged':True}))


if __name__=='__main__':
    main()
