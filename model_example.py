import torch
from se_attn.model import SelfExtendTransformer

# Create a random tensor of integers
x = torch.randint(0, 20000, (1, 1024))

# Initialize the BitNetTransformer
bitnet = SelfExtendTransformer(
    num_tokens=20000,
    dim=1024,
    depth=6,
    heads=8,
    ff_mult=4,
)

# Pass the tensor through the transformer
logits = bitnet(x)

# Print the shape of the output
print(logits.shape)
