"""Small decoder with genuine residual carry, RMSNorm, RoPE and SwiGLU.

An independent mechanism implementation, not the upstream MIT model/checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    widths: list[int]
    head_dim: int = 32
    seq_len: int = 128
    vocab_size: int = 256


def structure(widths: list[int]) -> list[dict]:
    """Track prior block participation, distinct from embedding availability."""
    last = [-1] * max(widths)
    records = []
    for layer, width in enumerate(widths):
        continuous, reactivated, new, ages = [], [], [], []
        for j in range(width):
            if last[j] == -1:
                new.append(j)
            elif last[j] == layer - 1:
                continuous.append(j)
            else:
                reactivated.append(j)
                ages.append(layer - 1 - last[j])
        records.append({"layer": layer, "width": width, "C": continuous,
                        "R": reactivated, "new": new, "R_ages": ages})
        for j in range(width):
            last[j] = layer
    return records


class Attention(nn.Module):
    def __init__(self, width: int, head_dim: int):
        super().__init__()
        if width % head_dim or head_dim % 2:
            raise ValueError("Widths must divide into even RoPE head dimensions")
        self.heads, self.head_dim = width // head_dim, head_dim
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.register_buffer("freq", 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)), persistent=False)

    def rope(self, x):
        angles = torch.outer(torch.arange(x.shape[-2], device=x.device), self.freq)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        z = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
        return self.o_proj(z.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        self.width = width
        self.attn_norm = nn.RMSNorm(width)
        self.attn = Attention(width, head_dim)
        self.mlp_norm = nn.RMSNorm(width)
        hidden = math.ceil((8 * width / 3) / 32) * 32
        self.up = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, h):
        x = h[..., :self.width]
        x = x + self.attn(self.attn_norm(x))
        z = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate(z)) * self.up(z))
        # The unused coordinates remain bitwise unchanged, including their history.
        return torch.cat((x, h[..., self.width:]), dim=-1)


class VariableWidthLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.widths[0] != max(config.widths) or config.widths[-1] != max(config.widths):
            raise ValueError("Pilot requires first/last blocks at maximum width; new dimensions need another protocol")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, max(config.widths))
        self.blocks = nn.ModuleList([Block(w, config.head_dim) for w in config.widths])
        self.final_norm = nn.RMSNorm(max(config.widths))
        self.lm_head = nn.Linear(max(config.widths), config.vocab_size, bias=False)
        # Untied output keeps initial language-model loss near log(vocabulary size).
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens):
        h = self.embedding(tokens)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.final_norm(h))

    def targets(self):
        for rec in structure(self.config.widths):
            if rec["R"]:
                for projection in ("q_proj", "v_proj"):
                    name = f"blocks.{rec['layer']}.attn.{projection}"
                    yield name, getattr(self.blocks[rec["layer"]].attn, projection), rec
