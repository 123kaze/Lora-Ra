import unittest
import torch
from torch.nn import functional as F

from carryrank.allocation import allocate
from carryrank.curvature import relative_diagonal,spectral_utility,whiten
from carryrank.model import ModelConfig,VariableWidthLM


class CurvatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)
        torch.set_num_threads(2)

    def test_relative_damping_two_sides_and_allocation_are_invariant(self):
        gradient=torch.randn(7,5,dtype=torch.float64)
        hin=torch.rand(5,dtype=torch.float64)+.01
        hout=torch.rand(7,dtype=torch.float64)+.001
        original=whiten(gradient,hin,hout)
        groups=[{'id':'Q','m':7,'n':5,'cost':12},{'id':'V','m':7,'n':5,'cost':12}]
        _,reference=spectral_utility(original,5)
        v=[x*.8 for x in reference]
        ranks=allocate(groups,{'Q':reference,'V':v},48,5)[0]
        for c in (.125,8.,3.7):
            updated=whiten(gradient/c,hin,hout/(c*c))
            self.assertTrue(torch.allclose(updated,original,rtol=1e-12,atol=1e-12))
            _,utility=spectral_utility(updated,5)
            self.assertEqual(allocate(groups,{'Q':utility,'V':v},48,5)[0],ranks)

    def test_fixed_absolute_output_damping_breaks_invariance(self):
        gradient=torch.randn(7,5,dtype=torch.float64)
        hin=torch.ones(5,dtype=torch.float64)
        hout=torch.ones(7,dtype=torch.float64)*.001
        epsilon=.01*hout.mean()
        original=gradient/(hout+epsilon).sqrt()[:,None]/relative_diagonal(hin).sqrt()
        changed=(gradient/8)/(hout/64+epsilon).sqrt()[:,None]/relative_diagonal(hin).sqrt()
        self.assertFalse(torch.allclose(original,changed,rtol=1e-3,atol=1e-6))

    def test_invalid_and_zero_curvature_is_explicit(self):
        for value in (torch.zeros(3),torch.tensor([1.,-1.,2.]),torch.tensor([1.,float('nan'),2.])):
            with self.assertRaises(ValueError):relative_diagonal(value)

    def test_measured_small_model_qk_transform(self):
        model=VariableWidthLM(ModelConfig([32,16,32],head_dim=8,seq_len=8)).double()
        tokens=torch.randint(0,256,(2,8))
        labels=torch.randint(0,256,(2,8))
        def measure():
            stats={}
            linear=model.blocks[2].attn.q_proj
            def hook(module,args,output):
                stats['hin']=args[0].detach().reshape(-1,32).square().mean(0)
                output.retain_grad();stats['output']=output
            handle=linear.register_forward_hook(hook)
            model.zero_grad(set_to_none=True)
            logits=model(tokens)
            F.cross_entropy(logits.flatten(0,1),labels.flatten()).backward()
            handle.remove()
            delta=stats['output'].grad.detach().reshape(-1,32)*labels.numel()
            return logits.detach(),linear.weight.grad.detach().clone(),stats['hin'],delta.square().mean(0)
        logits,g,hi,ho=measure()
        original=whiten(g,hi,ho)
        with torch.no_grad():
            model.blocks[2].attn.q_proj.weight.div_(8)
            model.blocks[2].attn.k_proj.weight.mul_(8)
        new_logits,new_g,new_hi,new_ho=measure()
        self.assertTrue(torch.allclose(logits,new_logits,rtol=1e-12,atol=1e-12))
        self.assertTrue(torch.allclose(whiten(new_g,new_hi,new_ho),original,rtol=1e-11,atol=1e-12))


if __name__=='__main__':unittest.main()
