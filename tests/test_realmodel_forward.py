"""API/gradient checks on synthetic tiny GPT-2, not experimental results."""
import unittest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model


class RealModelForwardTests(unittest.TestCase):
    def make_model(self, ada=False):
        torch.manual_seed(719)
        torch.set_num_threads(1)
        config = GPT2Config(vocab_size=37, n_embd=16, n_head=2, n_layer=2,
            n_positions=16, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., use_cache=False,
            bos_token_id=0, eos_token_id=0)
        config._attn_implementation = 'eager'
        base = GPT2LMHeadModel(config).requires_grad_(False)
        common = dict(task_type=TaskType.CAUSAL_LM, target_modules=['c_attn'], fan_in_fan_out=True)
        if ada:
            adapter = AdaLoraConfig(**common, init_r=2, target_r=1, total_step=4,
                tinit=1, tfinal=1, deltaT=1, orth_reg_weight=.5, lora_alpha=2)
        else:
            adapter = LoraConfig(**common, r=2, lora_alpha=1, use_rslora=True)
        model = get_peft_model(base, adapter).eval()
        if not ada:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if 'lora_B' in name:
                        parameter.normal_(0., .02)
        return model

    def inputs(self):
        ids = torch.tensor([[0, 0, 3, 4, 5, 6], [0, 9, 4, 3, 8, 7]])
        mask = ids.ne(0).long()
        return dict(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1)-1).clamp_min(0), use_cache=False)

    def test_last_position_loss_and_adapter_gradients_match_full_forward(self):
        model = self.make_model()
        targets = torch.tensor([11, 12]); inputs = self.inputs()
        full = model(**inputs).logits[:, -1]
        reference = F.cross_entropy(full, targets)
        reference.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        compact = model(**inputs, logits_to_keep=1, labels=targets[:, None], shift_labels=targets[:, None])
        torch.testing.assert_close(compact.logits[:, -1], full, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(compact.loss, reference, atol=1e-7, rtol=1e-6)
        compact.loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-7, rtol=1e-5)

    def test_adalora_orthogonal_regularizer_is_included(self):
        model = self.make_model(ada=True); targets = torch.tensor([11, 12])
        output = model(**self.inputs(), logits_to_keep=1, labels=targets[:, None], shift_labels=targets[:, None])
        data_loss = F.cross_entropy(output.logits[:, -1], targets)
        terms = []
        for name, parameter in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                covariance = parameter@parameter.T if 'lora_A' in name else parameter.T@parameter
                terms.append(torch.linalg.matrix_norm(covariance-torch.eye(len(covariance))))
        expected = .5*torch.stack(terms).mean()
        self.assertGreater(float(expected.detach()), 0.)
        torch.testing.assert_close(output.loss-data_loss, expected, atol=1e-6, rtol=1e-6)

    def test_left_padding_positions_preserve_last_token_prediction(self):
        model = self.make_model(); inputs = self.inputs()
        with torch.no_grad():
            padded = model(**inputs, logits_to_keep=1).logits[:, -1]
            for index in range(2):
                valid = inputs['input_ids'][index][inputs['attention_mask'][index].bool()][None]
                unpadded = model(input_ids=valid, use_cache=False, logits_to_keep=1).logits[:, -1]
                torch.testing.assert_close(padded[index:index+1], unpadded, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
