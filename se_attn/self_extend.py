import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from zeta.nn import YarnEmbedding


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


class SelfExtendAttn(nn.Module):
    """
    SelfExtendAttn module performs self-attention on input sequences using
    both normal self-attention and grouped self-attention.

    Args:
        dim (int): The dimension of the input sequences.
        g_size (int): The size of each group for grouped self-attention.
        w_size (int): The window size for grouped self-attention.
        qk_norm (bool, optional): Whether to apply layer normalization to
            query and key vectors. Defaults to False.

    Example:
    import torch
    from se_attn import SelfExtendAttn

    # Example usage
    dim = 512  # Dimension of model
    g_size = 2  # Group size
    w_size = 4  # Window size for neighbor tokens
    self_extend = SelfExtendAttn(dim, g_size, w_size, qk_norm=True)

    # Example tensors for q, k, v, and pos
    q = torch.randn(1, 10, dim)
    k = torch.randn(1, 10, dim)
    v = torch.randn(1, 10, dim)
    pos = torch.arange(0, 10).unsqueeze(0)  # Example positional indices

    output = self_extend(q, k, v, pos)
    print(output)

    """

    def __init__(
        self,
        dim: int,
        g_size: int,
        w_size: int,
        qk_norm: bool = False,
        yarn_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        super(SelfExtendAttn, self).__init__()
        self.dim = dim
        self.g_size = g_size
        self.w_size = w_size
        self.qk_norm = qk_norm
        self.yarn_embeddings = yarn_embeddings

        # Layer normalization
        if qk_norm:
            self.norm = nn.LayerNorm(dim)

        # Yarn embeddings
        if yarn_embeddings:
            self.yarn = YarnEmbedding(dim, *args, **kwargs)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, pos: Tensor
    ) -> Tensor:
        """
        Forward pass of the SelfExtendAttn module.

        Args:
            q (torch.Tensor): The query tensor of shape (batch_size, seq_len, dim).
            k (torch.Tensor): The key tensor of shape (batch_size, seq_len, dim).
            v (torch.Tensor): The value tensor of shape (batch_size, seq_len, dim).
            pos (torch.Tensor): The position tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, dim).
        """
        seq_len = q.size(1)

        # Apply layer normalization to query and key vectors
        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)

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
