"""Calibration-only two-sided diagonal proxy and measured Q/K invariance test."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from carryrank.allocation import allocate
from carryrank.curvature import relative_diagonal,spectral_utility,whiten
from carryrank.data import prepare,random_batch
from carryrank.experiment import ROOT,base_digest,code_manifest,digest,read_json,write_csv,write_json
from carryrank.model import ModelConfig,VariableWidthLM,structure


def relative_error(actual,expected):
    return (actual-expected).norm().item()/max(expected.norm().item(),1e-300)


def collect(model,data,config,reference_logits=None,reference_losses=None):
    model.eval();model.requires_grad_(False)
    targets=list(model.targets())
    stats={};handles=[]
    for name,linear,_ in targets:
        linear.weight.requires_grad_(True)
        stats[name]={'G':torch.zeros_like(linear.weight,device='cpu',dtype=torch.float64),
                     'Hin':torch.zeros(linear.in_features,dtype=torch.float64),
                     'Hout':torch.zeros(linear.out_features,dtype=torch.float64),'count':0,'backward_count':0}
        def capture(module,inputs,output,key=name):
            x=inputs[0].detach().reshape(-1,module.in_features)
            count=x.shape[0]
            stats[key]['Hin'].add_(x.double().square().sum(0).cpu())
            stats[key]['count']+=count
            def backward(gradient):
                # CE uses mean over predicted tokens. Undo its 1/N before E[delta^2].
                delta=gradient.detach().reshape(-1,module.out_features).double()*count
                stats[key]['Hout'].add_(delta.square().sum(0).cpu())
                stats[key]['backward_count']+=count
            output.register_hook(backward)
        handles.append(linear.register_forward_hook(capture))
    rng=torch.Generator().manual_seed(config['calibration']['seed'])
    batches=config['calibration']['batches']
    logits_cache=[];losses=[];max_logit_error=0.0;max_loss_error=0.0
    initial_hash=base_digest(model)
    for batch in range(batches):
        x,y=random_batch(data['target_train'],config['calibration']['batch_size'],config['model']['seq_len'],rng,'cuda')
        model.zero_grad(set_to_none=True)
        logits=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        if reference_logits is None:
            logits_cache.append(logits.detach().cpu())
        else:
            max_logit_error=max(max_logit_error,(logits.detach().cpu()-reference_logits[batch]).abs().max().item())
            max_loss_error=max(max_loss_error,abs(loss.item()-reference_losses[batch]))
        losses.append(loss.item())
        loss.backward()
        for name,linear,_ in targets:
            stats[name]['G'].add_(linear.weight.grad.detach().cpu().double()/batches)
    for handle in handles:handle.remove()
    for value in stats.values():
        assert value['count']==value['backward_count']==batches*config['calibration']['batch_size']*config['model']['seq_len']
        value['Hin']/=value['count'];value['Hout']/=value['count']
    assert base_digest(model)==initial_hash
    return stats,logits_cache,losses,max_logit_error,max_loss_error


def run(output):
    if output.exists():raise FileExistsError('Use a fresh calibration run id')
    if output.parent != ROOT/'runs':raise ValueError('Expected runs/<run-id>')
    parent=ROOT/'runs/diagnostic_20261122_02'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    settings={'base_seeds':[2026,2027,2028],'scales':[1.0,0.125,8.0],'damping_in':.01,'damping_out':.01,
              'budgets':[1024,3840],'max_rank':16,'schemes':['true','shuffled'],
              'metrics':['raw','diag_input','diag_both'],'training_performed':False,
              'Hout_definition':'E[(N * d(mean_token_loss)/d(projection_output))^2]',
              'data_split':'target_train only'}
    output.mkdir(parents=True)
    write_json(output/'config.json',settings)
    sources=code_manifest();sources['scripts/calibrate_output_diagonal.py']=digest(Path(__file__).read_bytes())
    write_json(output/'identity.json',{'source':sources,'protocol_sha256':digest((ROOT/'plan/output-diagonal-protocol.md').read_bytes()),
               'parent_base_files':{str(s):digest((parent/f'base_{s}/base.pt').read_bytes()) for s in settings['base_seeds']}})
    shutil.copy2(ROOT/'plan/output-diagonal-protocol.md',output/'protocol.md')
    spectral=[];invariance=[];allocations=[];moment_rows=[];references={}
    for seed in settings['base_seeds']:
        child=parent/f'base_{seed}'
        config=read_json(child/'config.json')
        splits,data_manifest=prepare(ROOT,config['data'])
        assert data_manifest==read_json(child/'data_manifest.json')
        base=torch.load(child/'base.pt',map_location='cpu',weights_only=True)
        calibration=torch.load(child/'calibration.pt',map_location='cpu',weights_only=True)
        groups=calibration['groups']
        write_json(output/f'base_{seed}/groups.json',groups)
        reference_stats=None;reference_logits=None;reference_losses=None
        for scale in settings['scales']:
            model=VariableWidthLM(ModelConfig(**config['model'])).cuda()
            model.load_state_dict(base['model'])
            assert base_digest(model)==base['sha256']
            with torch.no_grad():
                for name,_,record in model.targets():
                    if name.endswith('q_proj'):
                        model.blocks[record['layer']].attn.q_proj.weight.mul_(scale)
                        model.blocks[record['layer']].attn.k_proj.weight.div_(scale)
            stats,cache,losses,logit_error,loss_error=collect(model,splits,config,reference_logits,reference_losses)
            if scale==1:
                reference_stats,reference_logits,reference_losses=stats,cache,losses
            arrays={};mapping={}
            for name,values in stats.items():
                key=name.replace('.','_')
                mapping[name]={component:component+'__'+key for component in ('G','Hin','Hout')}
                for component in ('G','Hin','Hout'):arrays[mapping[name][component]]=values[component].numpy()
                ref=reference_stats[name];factor=scale if name.endswith('q_proj') else 1.0
                moment_rows.append({'base_seed':seed,'q_multiplier':scale,'module':name,
                    'input_second_moment_mean':values['Hin'].mean().item(),
                    'output_second_moment_mean':values['Hout'].mean().item(),
                    'gradient_transform_relative_error':relative_error(values['G'],ref['G']/factor),
                    'input_transform_relative_error':relative_error(values['Hin'],ref['Hin']),
                    'output_transform_relative_error':relative_error(values['Hout'],ref['Hout']/(factor*factor)),
                    'max_logit_error_all_calibration_batches':logit_error,'max_loss_error':loss_error})
            token=str(scale).replace('.','p')
            np.savez_compressed(output/f'base_{seed}/scale_{token}.npz',**arrays)
            write_json(output/f'base_{seed}/scale_{token}.json',{'modules':mapping,'base_sha256':base['sha256'],
                       'q_multiplier':scale,'dtype':'float64','model_weights_included':False})
            for scheme in settings['schemes']:
                for metric in settings['metrics']+['fixed_absolute_out_control']:
                    utilities={}
                    for group in groups[scheme]:
                        name=group['module'];values=stats[name]
                        g=values['G'][:,group['indices']];hi=values['Hin'];ho=values['Hout']
                        if metric=='raw':matrix=g
                        elif metric=='diag_input':matrix=g/relative_diagonal(hi,.01)[group['indices']].sqrt()
                        elif metric=='diag_both':
                            matrix=whiten(g,hi[group['indices']],ho,.01,.01,input_reference_mean=hi.mean())
                        else:
                            fixed_epsilon=.01*reference_stats[name]['Hout'].mean()
                            matrix=g/(ho+fixed_epsilon).sqrt()[:,None]/relative_diagonal(hi,.01)[group['indices']].sqrt()
                        singular,utility=spectral_utility(matrix,16)
                        key=(seed,scheme,metric,group['id'])
                        if scale==1:references[key]=(matrix.clone(),singular.clone(),torch.tensor(utility,dtype=torch.float64))
                        ref_matrix,ref_singular,ref_utility=references[key]
                        invariance.append({'base_seed':seed,'q_multiplier':scale,'scheme':scheme,'metric':metric,
                            'group':group['id'],'matrix_relative_error':relative_error(matrix,ref_matrix),
                            'spectrum_relative_error':relative_error(singular,ref_singular),
                            'prefix_utility_relative_error':relative_error(torch.tensor(utility,dtype=torch.float64),ref_utility)})
                        utilities[group['id']]=utility
                        for rank,sigma in enumerate(singular.tolist(),1):
                            spectral.append({'base_seed':seed,'q_multiplier':scale,'scheme':scheme,'metric':metric,
                                'group':group['id'],'rank':rank,'sigma':sigma,'prefix_utility':utility[rank],'cost':group['cost']})
                    for budget in settings['budgets']:
                        ranks,spent,value=allocate(groups[scheme],utilities,budget,16)
                        key=(seed,scheme,metric,budget)
                        if scale==1:references[key]=ranks
                        q=sum(g['cost']*ranks[g['id']] for g in groups[scheme] if g['module'].endswith('q_proj'))
                        allocations.append({'base_seed':seed,'q_multiplier':scale,'scheme':scheme,'metric':metric,
                            'budget':budget,'spent':spent,'q_parameters':q,'v_parameters':spent-q,
                            'utility':value,'allocation_matches_c1':ranks==references[key],'ranks':ranks})
            del model
        print(f'Finished measured output-diagonal calibration for base {seed}',flush=True)
    both=[r for r in invariance if r['metric']=='diag_both']
    both_alloc=[r for r in allocations if r['metric']=='diag_both']
    checks={'all_calibration_logits_invariant':max(r['max_logit_error_all_calibration_batches'] for r in moment_rows)<=1e-5,
            'empirical_gradient_transform':max(r['gradient_transform_relative_error'] for r in moment_rows)<=1e-8,
            'empirical_output_moment_transform':max(r['output_transform_relative_error'] for r in moment_rows)<=1e-8,
            'two_sided_matrix_invariant':max(r['matrix_relative_error'] for r in both)<=1e-8,
            'two_sided_spectrum_invariant':max(r['spectrum_relative_error'] for r in both)<=1e-8,
            'two_sided_utility_invariant':max(r['prefix_utility_relative_error'] for r in both)<=1e-8,
            'two_sided_allocations_identical':all(r['allocation_matches_c1'] for r in both_alloc),
            'exact_budgets':all(r['spent']==r['budget'] for r in allocations)}
    identity=read_json(output/'identity.json')
    checks['parent_base_files_unchanged']=all(digest((parent/f'base_{s}/base.pt').read_bytes())==identity['parent_base_files'][str(s)] for s in settings['base_seeds'])
    if not all(checks.values()):raise AssertionError(checks)
    write_csv(output/'moments.csv',moment_rows);write_csv(output/'spectra.csv',spectral);write_csv(output/'invariance.csv',invariance)
    write_json(output/'allocations.json',allocations)
    write_csv(output/'allocation_summary.csv',[{k:v for k,v in row.items() if k!='ranks'} for row in allocations])
    result={'checks':checks,'pass':True,'training_steps':0,'actual_forward_backward_batches':3*3*12,
            'max_both_matrix_relative_error':max(r['matrix_relative_error'] for r in both),
            'max_both_spectrum_relative_error':max(r['spectrum_relative_error'] for r in both),
            'max_both_utility_relative_error':max(r['prefix_utility_relative_error'] for r in both),
            'two_sided_allocation_checks':len(both_alloc),
            'max_fixed_absolute_damping_matrix_error':max(r['matrix_relative_error'] for r in invariance if r['metric']=='fixed_absolute_out_control')}
    write_json(output/'verification.json',result)
    history=structure(config['model']['widths'])
    write_json(output/'structure_and_budget_feasibility.json',{'reactivation_ages':[
        {'layer_0based':r['layer'],'width':r['width'],'R_age_values':sorted(set(r['R_ages']))} for r in history if r['R']],
        'group_costs':[{k:g[k] for k in ('id','m','n','cost')} for g in groups['true']],
        'minimum_rank_cost':min(g['cost'] for g in groups['true']),
        'last_V_C_rank1_cost':next(g['cost'] for g in groups['true'] if g['id']=='blocks.4.attn.v_proj/C')})
    destination=ROOT/'results'/output.name
    if destination.exists():raise FileExistsError(destination)
    shutil.copytree(output,destination)
    for relative,expected in sources.items():
        target=destination/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);assert digest(target.read_bytes())==expected
    files=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())} for p in destination.rglob('*') if p.is_file()]
    write_json(destination/'artifact-manifest.json',{'run':output.name,'files':files})
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output.resolve())
