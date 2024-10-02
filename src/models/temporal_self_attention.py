import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class TemporalSelfAttention(nn.Module):
    """
    Implementation of TemporalSelfAttention.
    """
    def __init__(self, in_features, out_features, n, s=20.0, m=0.5, easy_margin=False):
        super.__init__()

        self.num_heads = 8
        self.embed_dim = 256

        self.temporal_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query, prev_embed=None):
        bypass = query
        if prev_embed is None:
            prev_embed = query
        output, _ = self.temporal_attention(query, prev_embed, prev_embed)
        output = self.norm(output+bypass)
        return output

