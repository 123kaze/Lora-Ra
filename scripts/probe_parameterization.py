"""Post-hoc Q/K reparameterization witness; unchanged function, no training.

Wq -> c Wq, Wk -> Wk/c preserves attention scores, while Gq -> Gq/c.
This tests the coordinate dependence of raw and input-only rank utilities.
"""
import argparse

import torch
from torch.nn import functional as F

from carryrank.allocation import allocate
from carryrank.data import prepare,random_batch
from carryrank.experiment import ROOT,base_digest,digest,read_json,write_csv,write_json
from carryrank.model import ModelConfig,VariableWidthLM


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=ROOT/args.output
    if output.exists(): raise FileExistsError(output)
    parent=ROOT/'runs/pilot_20261122_01'
    cfg=read_json(parent/'config.json')
    base=torch.load(parent/'base.pt',map_location='cpu',weights_only=True)
    calibration=torch.load(parent/'calibration.pt',map_location='cpu',weights_only=True)
    moments={k:torch.tensor(v,dtype=torch.float64) for k,v in read_json(ROOT/'results/calibration_probe_20261122_01/input_second_moments.json').items()}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    data,_=prepare(ROOT,cfg['data'])
    rng=torch.Generator().manual_seed(cfg['calibration']['seed'])
    x,y=random_batch(data['target_train'],16,128,rng,'cuda')
    original_logits=None;reference_gradients={};rows=[]
    for scale in (1.0,0.125,8.0):
        model=VariableWidthLM(ModelConfig(**cfg['model'])).cuda()
        model.load_state_dict(base['model']);model.eval();model.requires_grad_(False)
        with torch.no_grad():
            for name,_,record in model.targets():
                if name.endswith('q_proj'):
                    attention=model.blocks[record['layer']].attn
                    attention.q_proj.weight.mul_(scale)
                    attention.k_proj.weight.div_(scale)
        targets=list(model.targets())
        for _,linear,_ in targets: linear.weight.requires_grad_(True)
        logits=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        loss.backward()
        if scale==1.0:
            original_logits=logits.detach().clone()
            reference_gradients={name:linear.weight.grad.detach().cpu().double() for name,linear,_ in targets}
        output_error=(logits-original_logits).abs().max().item()
        gradient_errors=[]
        for name,linear,_ in targets:
            expected=reference_gradients[name]/scale if name.endswith('q_proj') else reference_gradients[name]
            actual=linear.weight.grad.detach().cpu().double()
            gradient_errors.append((actual-expected).norm().item()/max(expected.norm().item(),1e-30))
        for metric in ('raw','diag_input'):
            utilities={}
            for group in calibration['groups']['true']:
                name=group['module'];matrix=calibration['gradients'][name][:,group['indices']]
                if name.endswith('q_proj'): matrix=matrix/scale
                if metric=='diag_input':
                    moment=moments[name]
                    matrix=matrix/(moment[group['indices']]+.01*moment.mean()).sqrt()
                singular=torch.linalg.svdvals(matrix)[:16]
                utilities[group['id']]=[0.0]+(.5*singular.square()).cumsum(0).tolist()
            for budget in (1024,3840):
                ranks,spent,utility=allocate(calibration['groups']['true'],utilities,budget,16)
                qcost=sum(g['cost']*ranks[g['id']] for g in calibration['groups']['true'] if g['module'].endswith('q_proj'))
                rows.append({'q_multiplier':scale,'metric':metric,'budget':budget,'q_parameters':qcost,
                             'v_parameters':budget-qcost,'predicted_utility':utility,
                             'max_logit_error_vs_original':output_error,'max_gradient_transform_relative_error':max(gradient_errors),
                             'training_batch_loss':loss.item(),'ranks':str(ranks)})
        del model,logits
    assert max(r['max_logit_error_vs_original'] for r in rows)<1e-5
    assert max(r['max_gradient_transform_relative_error'] for r in rows)<1e-5
    output.mkdir(parents=True)
    write_csv(output/'parameterization.csv',rows)
    write_json(output/'provenance.json',{'parent_base_sha256':base['sha256'],'model_training_performed':False,
        'data':'one fixed training calibration batch; full saved mean gradients transformed analytically',
        'operation':'Wq *= c; Wk /= c on expansion layers only',
        'limitation':'demonstrates coordinate dependence, not proof that Q capacity improves generalization',
        'script_sha256':digest(__import__('pathlib').Path(__file__).read_bytes())})
    print(__import__('json').dumps(rows,indent=2))


if __name__=='__main__':main()
