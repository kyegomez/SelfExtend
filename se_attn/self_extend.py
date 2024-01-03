import math
import torch
import torch.nn.functional as F
from torch import nn


def causal_mask(size):
    return torch.tril(torch.ones(size, size)).type(torch.bool)


def positional_encoding(seq_len, dim):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def apply_pos_emcode(tensor, pos, dim):
    seq_len = tensor.size(1)
    pe = positional_encoding(seq_len, dim)
    pe = pe.unsqueeze(0)
    return tensor + pe[:, pos]


class SelfExtend(nn.Module):
    def __init__(
        self,
        dim: int,
        g_size: int,
        w_size: int,
        qk_norm: bool = False,
    ):
        super(SelfExtend, self).__init__()
        self.dim = dim
        self.g_size = g_size
        self.w_size = w_size

        if qk_norm:
            self.norm = nn.LayerNorm(dim)

    def forward(self, q, k, v, pos):
        seq_len = q.size(1)

        # Normal self-attention for neighbor tokens
        ngb_q = apply_pos_emcode(q, pos, self.dim)
        ngb_k = apply_pos_emcode(k, pos, self.dim)
        ngb_attn = torch.matmul(ngb_q, ngb_k.transpose(-2, -1))
        ngb_attn = ngb_attn.masked_fill(
            ~causal_mask(seq_len), float("-inf")
        )

        # Grouped self-attention
        g_pos = pos // self.g_size
        shift = self.w_size - self.w_size // self.g_size
        s_g_pos = g_pos + shift
        g_q = apply_pos_emcode(q, s_g_pos, self.dim)
        g_k = apply_pos_emcode(k, g_pos, self.dim)
        g_attn = torch.matmul(g_q, g_k.transpose(-2, -1))
        g_attn = g_attn.masked_fill(
            ~causal_mask(seq_len), float("-inf")
        )

        # Create masks for merging attentions
        g_mask = torch.tril(
            torch.ones(seq_len - self.w_size, seq_len - self.w_size)
        )
        mask = torch.ones(seq_len, seq_len)
        mask[self.w_size :, : -self.w_size] -= g_mask

        # Merge attentions
        attn = torch.where(mask.bool(), ngb_attn, g_attn)
        attn_weights = F.softmax(attn, dim=-1)

        # Output
        output = torch.matmul(attn_weights, v)
        return output


# Example usage
dim = 512  # Dimension of model
g_size = 2  # Group size
w_size = 4  # Window size for neighbor tokens
self_extend = SelfExtend(dim, g_size, w_size)

# Example tensors for q, k, v, and pos
q = torch.randn(1, 10, dim)
k = torch.randn(1, 10, dim)
v = torch.randn(1, 10, dim)
pos = torch.arange(0, 10).unsqueeze(0)  # Example positional indices

output = self_extend(q, k, v, pos)
print(output)
