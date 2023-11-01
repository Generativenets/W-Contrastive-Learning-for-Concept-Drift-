import torch.nn as nn
import torch.nn.functional as F
from Attention import MultiheadAttention
import torch



class SetTransformer(nn.Module):
    def __init__(self, dim, n_heads, n_layers):
        super(SetTransformer, self).__init__()

        self.layers = nn.ModuleList([
            MultiheadAttention(dim, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x: batch_size*set_size*dim
        x = torch.mean(x, dim=1)
        return x  # x:batch_size*dim


