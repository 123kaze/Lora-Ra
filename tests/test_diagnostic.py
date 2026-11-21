import itertools
import unittest

import torch

from carryrank.allocation import allocate
from carryrank.diagnostic import rank_scale,reserve_q
from carryrank.lora import GroupedLinear


class DiagnosticTests(unittest.TestCase):
    def test_q_floor_matches_exhaustive_search(self):
        groups=[{'id':f'{module}/{label}','module':module,'cost':cost,'m':2,'n':2}
                for module,cost in [('a.q_proj',3),('b.q_proj',4),('b.v_proj',2)] for label in ('C','R')]
        utilities={g['id']:[0,10-i,15-i] for i,g in enumerate(groups)}
        budget=17
        allocation,spent,value=reserve_q(groups,utilities,budget,2)
        brute=[]
        for combination in itertools.product(range(3),repeat=6):
            if sum(g['cost']*k for g,k in zip(groups,combination))==budget and sum(combination[:2])>=1 and sum(combination[2:4])>=1:
                brute.append(sum(utilities[g['id']][k] for g,k in zip(groups,combination)))
        self.assertEqual(spent,budget)
        self.assertEqual(value,max(brute))
        self.assertTrue(all(k<=2 for k in allocation.values()))

    def test_scaling_reference_and_merge(self):
        self.assertEqual(rank_scale('rs_ref3',3),1.0)
        layer=GroupedLinear(torch.nn.Linear(8,6,bias=False),[(list(range(4)),2),([4,5,6,7],4)])
        for factor in layer.factors:
            factor.scaling=rank_scale('rs_ref3',factor.A.shape[0])
            with torch.no_grad(): factor.B.normal_()
        x=torch.randn(2,5,8)
        self.assertTrue(torch.allclose(layer(x),layer.merged()(x),atol=1e-6))

    def test_input_whitening_removes_pure_column_rescaling_without_damping(self):
        gradient=torch.randn(7,5,dtype=torch.float64)
        moment=torch.rand(5,dtype=torch.float64)+.1
        scale=torch.tensor([.2,1.,3.,.7,5.],dtype=torch.float64)
        original=torch.linalg.svdvals(gradient/moment.sqrt())
        transformed=torch.linalg.svdvals((gradient*scale)/(moment*scale.square()).sqrt())
        self.assertTrue(torch.allclose(original,transformed,atol=1e-12))


if __name__=='__main__': unittest.main()
