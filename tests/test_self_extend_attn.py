import pytest
import torch
from se_attn.self_extend import SelfExtendAttn


def test_self_extend_init():
    dim = 512
    g_size = 2
    w_size = 4
    self_extend = SelfExtendAttn(dim, g_size, w_size)
    assert self_extend.dim == dim
    assert self_extend.g_size == g_size
    assert self_extend.w_size == w_size


def test_self_extend_forward():
    dim = 512
    g_size = 2
    w_size = 4
    self_extend = SelfExtendAttn(dim, g_size, w_size)
    q = torch.randn(1, 10, dim)
    k = torch.randn(1, 10, dim)
    v = torch.randn(1, 10, dim)
    pos = torch.arange(0, 10).unsqueeze(0)
    output = self_extend(q, k, v, pos)
    assert output.shape == torch.Size([1, 10, dim])


@pytest.mark.parametrize(
    "dim, g_size, w_size", [(512, 2, 4), (256, 3, 5), (128, 4, 6)]
)
def test_self_extend_with_different_params(dim, g_size, w_size):
    self_extend = SelfExtendAttn(dim, g_size, w_size)
    q = torch.randn(1, 10, dim)
    k = torch.randn(1, 10, dim)
    v = torch.randn(1, 10, dim)
    pos = torch.arange(0, 10).unsqueeze(0)
    output = self_extend(q, k, v, pos)
    assert output.shape == torch.Size([1, 10, dim])
