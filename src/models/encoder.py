import torch
import torch.nn as nn
from models.temporal_self_attention import TemporalSelfAttention
from models.spatial_cross_attention import SpatialCrossAttention

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = 256    # self.config["embed_dim"]
        self.num_query = 1      # self.config["num_query"]
        self.num_heads = 16      # self.config["num_heads"]

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
        B, num_imgs, channels, h, w = x.shape # [32, 10, 256, 12, 26]
        ref_2d = self.get_reference_points(h, w, B)
        all_feats = x.flatten(3).permute(0, 1, 3, 2)  # [32, 10, 312, 256]
        query = query.unsqueeze(0).repeat(B, 1, 1) # [32, 1, 256]
        _,num_imgs,resolu,_ = all_feats.shape

        output = None
        for i in range(num_imgs):
            single_feat = all_feats[:, i, :, :].view(B, resolu, self.num_heads, -1)
            print("Single Feat Size:", single_feat.shape) # [32, 312, 16, 16]

            output = self.tsa(query, self.prev_embed)
            print("TSA Size:", output.shape) # [32, 1, 256]

            output = self.sca(output, single_feat)
            output = output.mean(1).unsqueeze(1)
            output = self.mlp(output) + output
            output = self.norm(output)
            self.prev_embed = output

        self.prev_embed = None

        return output

    def get_reference_points(self, H=4, W=11, bs=8, device='mps'):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float, device=device))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d
