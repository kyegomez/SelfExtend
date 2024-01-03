import torch
from se_attn import SelfExtend

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
