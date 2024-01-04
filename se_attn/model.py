import torch
from torch import nn, Tensor
from se_attn.self_extend import SelfExtendAttn
from zeta.nn import RMSNorm, FeedForward

# helpers


# [TRANSFORMER] Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        ff_mult: int,
        g_size: int,
        w_size: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.ff_mult = ff_mult
        self.g_size = g_size
        self.w_size = w_size

        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                SelfExtendAttn(
                    dim,
                    g_size,
                    w_size,
                    True,
                )
            )

            self.ffn_layers.append(
                FeedForward(dim, dim, ff_mult, *args, **kwargs),
            )

    def forward(self, x: Tensor):
        for attn, ffn in zip(self.layers, self.ffn_layers):
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            x = attn(x, x, x, pos) + x
            x = ffn(x) + x
        return x


# [MAIN MODEL] SelfExtendTransformer
class SelfExtendTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_tokens: int,
        heads=8,
        ff_mult=4,
        g_size: int = 2,
        w_size: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_tokens = num_tokens
        self.heads = heads
        self.ff_mult = ff_mult
        self.g_size = g_size
        self.w_size = w_size

        assert self.g_size is not None
        assert self.w_size is not None

        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            g_size=g_size,
            w_size=w_size,
        )

        self.to_logits = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, num_tokens)
        )

    def forward(self, x: Tensor):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
