"""Independently reconstruct all final phase-2 adapters and verify exported hashes."""
import argparse
import json
from pathlib import Path

import torch

from carryrank.diagnostic import rank_scale
from carryrank.experiment import ROOT, Experiment, base_digest, digest, read_json, write_json
from carryrank.lora import GroupedLinear, attach


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    output=args.run.resolve()
    cfg=read_json(output/'diagnostic_config.json')
    errors=[]
    output_shifts=[]
    budgets=[]
    for child in sorted(output.glob('base_*')):
        experiment=Experiment(read_json(child/'config.json'),child,resume=True)
        for result_file in sorted(child.glob('diagnostic_adapters/*/result.json')):
            result=read_json(result_file)
            final=result['evaluations'][-1]
            saved=torch.load(result_file.parent/'adapter.pt',map_location='cpu',weights_only=True)
            model,base=experiment.load_base()
            probe=experiment.splits['target_train'][:experiment.model_config.seq_len][None].to(experiment.device)
            model.eval()
            with torch.no_grad():
                original_logits=model(probe)
            count=attach(model,saved['groups'],saved['ranks'],1.0)
            for module in model.modules():
                if isinstance(module,GroupedLinear):
                    for factor in module.factors:
                        factor.scaling=rank_scale(saved['mode'],factor.A.shape[0])
            state=model.state_dict()
            assert set(saved['adapter'])=={key for key in state if '.factors.' in key}
            state.update(saved['adapter'])
            model.load_state_dict(state,strict=True)
            assert base_digest(model)==saved['base_sha256']==base['sha256']
            assert count==final['budget']
            model.eval()
            with torch.no_grad():
                change=(model(probe)-original_logits).float().square().mean().sqrt().item()
            output_shifts.append({'base_seed':final['base_seed'],'adapter_seed':final['adapter_seed'],
                'regime':final['regime'],'method':final['method'],'mode':final['mode'],'lr':final['lr'],
                'step':final['step'],'probe_logit_rms_change':change})
            value=experiment.evaluate(model,'target_val')['loss']
            error=abs(value-final['val_loss'])
            assert error<1e-5,(result_file,error)
            errors.append(error)
            budgets.append(count)
            del model
    published=ROOT/'results'/output.name
    manifest=read_json(published/'artifact-manifest.json')
    for entry in manifest['files']:
        assert digest((ROOT/entry['path']).read_bytes())==entry['sha256']
    original=ROOT/cfg['parent_run']
    assert digest((original/'base.pt').read_bytes())==read_json(output/'diagnostic_identity.json')['parent_base_file_sha256']
    result={'run':output.name,'checkpoints_reconstructed':len(errors),
            'max_validation_loss_error':max(errors),'exported_hashes_verified':len(manifest['files']),
            'parent_base_unchanged':True,'pass':True}
    write_json(output/'checkpoint_audit.json',result)
    write_json(published/'checkpoint_audit.json',result)
    from carryrank.experiment import write_csv
    write_csv(published/'final_output_shifts.csv',output_shifts)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
