import torch.nn as nn
from models import TemporalSelfAttention, SpatialCrossAttention

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = 256    # self.config["embed_dim"]
        self.num_query = 1      # self.config["num_query"]
        self.num_heads = 8      # self.config["num_heads"]

        # main components
        self.tsa = TemporalSelfAttention()
        self.sca = SpatialCrossAttention()
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        # history feature embedding, E_{i-1}
        self.prev_embed = None

    def forward(self, query, x):
        B, _, _, _, _ = x.shape
        query = query.unsqueeze(0).repeat(B, 1, 1)
        all_feats = x.flatten(3).permute(0, 1, 3, 2)  # [8,10,120,256]
        _, num_imgs, _, _ = all_feats.shape
        output = None

        for i in range(num_imgs):
            output = self.tsa(query, self.prev_embed)
            output = self.sca(output)
            output = output.mean(1).unsqueeze(1)
            output = self.mlp(output) + output
            output = self.norm(output)
            self.prev_embed = output

        self.prev_embed = None

        return output
