import torch.nn as nn
from deformable_attention import DeformableAttention

class SpatialCrossAttention(nn.Module):
    """
    Alternate implementation of SpatialCrossAttention using DAT rather than DETR attention module.
    """
    def __init__(self):
        super().__init__()
        self.deformable_attention = DeformableAttention(
            dim = 256,                      # feature dimensions
            dim_head = 16,                  # dimension per head
            heads = 16,                     # attention heads
            dropout = 0.1,                   # dropout
            downsample_factor = 4,          # downsample factor (r in paper)
            offset_scale = 4,               # scale of offset, maximum offset
            offset_groups = 4,              # number of offset groups, a factor of heads
            offset_kernel_size = 6,         # offset kernel size
        )
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        output = self.deformable_attention(x) + x
        output = self.norm(output)
        return output
