from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # two linear projections, sandwiching a GELU nonlinearity
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)  # Layer norm
        self.attn = CausalSelfAttention(config)  # Attention
        self.ln_2 = nn.LayerNorm(config.n_embd)  # Layer norm
        self.mlp = MLP(config)  # MLP

    # repeated map-reduce
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # reduce
        x = x + self.mlp(self.ln_2(x))  # map
        return x


class GPT(nn.module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings:
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Positional embeddings:
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # Hidden layers:
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Final layer norm:
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
