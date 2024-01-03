[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# SelfExtendAttn
Implementation of SelfExtendAttn from the paper "LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning" from Pytorch and Zeta. This implementation is based mostly on the pseudocode listed in Algorithm 1 in page 4


# Install
`pip install SelfExtendAttn`


## Usage
```python
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
```

Certainly! A technical architecture analysis of the `SelfExtendAttn` attention mechanism for a README.md file would involve detailing the purpose, design, and usage of the module. Here's a suggested layout and content:

---

## Technical Architecture

### Key Concepts

- **Grouped Attention**: This mechanism divides the input sequence into groups and applies the attention operation within each group. It uses a floor operation to adjust the positions within the groups, enabling efficient handling of longer sequences.
  
- **Normal Attention**: Standard self-attention used in transformers, focusing on nearby tokens within a specified window.

### Attention Mechanism

The `SelfExtendAttn` module integrates these two attention strategies:

1. **Normal Attention** is applied to tokens within a neighborhood window, maintaining precise positional information for closely related tokens.
   
2. **Grouped Attention** is used for tokens outside this neighborhood window. It reduces the granularity of positional information for distant tokens, which is less critical but still contributes to the overall context understanding.

### Merge Strategy

The attention values outside the neighborhood window are replaced by those obtained from the grouped attention. This merging strategy ensures a smooth transition and efficient processing of longer sequences while preserving the essential context captured by the normal attention within the neighborhood window.

### Positional Encoding

Sine and cosine functions generate positional encodings, ensuring that the model retains an understanding of token order and position.

## Implementation Details

- **Module Class**: `SelfExtendAttn` is implemented as a subclass of `nn.Module` in PyTorch.
- **Configurability**: Key parameters such as group size and neighbor window size are configurable.
- **Causal Masking**: Ensures that the attention mechanism respects the autoregressive property of language models.



# Citation
```bibtext
@misc{jin2024llm,
    title={LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning}, 
    author={Hongye Jin and Xiaotian Han and Jingfeng Yang and Zhimeng Jiang and Zirui Liu and Chia-Yuan Chang and Huiyuan Chen and Xia Hu},
    year={2024},
    eprint={2401.01325},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# License
MIT