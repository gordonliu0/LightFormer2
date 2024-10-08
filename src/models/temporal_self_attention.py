import torch.nn as nn

class TemporalSelfAttention(nn.Module):
    """
    Implementation of TemporalSelfAttention.
    """
    def __init__(self):
        super().__init__()

        self.num_heads = 8
        self.embed_dim = 256

        self.mha = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query, prev_embed=None):
        if prev_embed is None:
            prev_embed = query
        output, _ = self.mha(query, prev_embed, prev_embed)
        output = self.norm(output + query)
        print("TSA shape:", output.shape)
        return output

