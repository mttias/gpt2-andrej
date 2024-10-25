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


class CausalSelfAttention(nn.Modue):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # weights. once we multiply our input (n_embd * 1) with (3*n_embd x n_embd),
        # we get a (3*n_embd * 1) matrix.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Weight matrix is (3*n_embd, n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a bias, more of a mask, but following the oAI/HF naming
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        """
        x: (batch_size, seq_length, n_embd)
        """
        B, T, C = x.size()  # (batch_size, seq_length, n_embd)

        # the c_attn (lin. proj) operates on the last dimension of x,
        # i.e. (n_embd * 1) x (3*n_embd x n_embd) -> (3*n_embd * 1)
        # splits into q, k, v (n_embd, n_embd) each
        # dim=2 refers to the third dimension
        # this splits the (batch_size, seq_length, 3*n_embd) matrix into
        # three different 3d tensors (batch_size, seq_length, n_embd)
        # n_embd_q is query, n_embd_k is key, n_embd_v is value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # points to same range in memory (contiguous), but reshapes
        # self.n_head = 12, C = 768, so we're splitting into 12 heads of 64 dim
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # matrix multiplication of q and k.
        # then we scale by K = 1/sqrt(k.size(-1))
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask = hide future tokens
        # the mask is a lower triangular matrix, so we set the attention scores to -inf
        # for the tokens beyond the actual sequence length.
        # example: if T = 3, then the mask is:
        # [[[1, 0, 0],
        #   [1, 1, 0],
        #   [1, 1, 1]]]
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # finally, we take the softmax
        # softmax turns -inf into 0
        attn = F.softmax(attn, dim=-1)

        y = attn @ v  # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y


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
