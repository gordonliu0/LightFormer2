
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from deformable_attention import DeformableAttention
from deformable_attention.deformable_attention_2d import create_grid_like, normalize_grid
from einops import rearrange
from functools import partial

def _forward(self, x, q = None, return_vgrid = False):
    """
    b - batch
    h - heads
    x - height
    y - width
    d - dimension
    g - offset groups
    """
    heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

    # queries
    if q == None:
        q = self.to_q(x)
    else:
        bs, _, dim = q.shape
        q = q.repeat(1, 1, h*w)
        q = q.view(bs, 1, dim, h, w)
        q = q.squeeze(1)
    group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)
    grouped_queries = group(q)
    offsets = self.to_offsets(grouped_queries)

    # calculate grid + offsets
    grid = create_grid_like(offsets)
    vgrid = grid + offsets
    vgrid_scaled = normalize_grid(vgrid)
    kv_feats = F.grid_sample(
        group(x),
        vgrid_scaled,
    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
    kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

    # derive key / values
    k, v = self.to_k(kv_feats), self.to_v(kv_feats)

    # scale queries
    q = q * self.scale

    # split out heads
    q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

    # query / key similarity
    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    # relative positional bias
    grid = create_grid_like(x)
    grid_scaled = normalize_grid(grid, dim = 0)
    rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
    sim = sim + rel_pos_bias

    # numerical stability
    sim = sim - sim.amax(dim = -1, keepdim = True).detach()

    # attention
    attn = sim.softmax(dim = -1)
    attn = self.dropout(attn)

    # aggregate and combine heads
    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
    out = self.to_out(out)

    if return_vgrid:
        return out, vgrid

    return out

class SpatialCrossAttention(nn.Module):
    """
    Alternate implementation of SpatialCrossAttention using DAT rather than DETR attention module.
    """
    def __init__(self):
        super().__init__()

        # DAT adapted for TSA query input (TSA query repeated to attend to each input 'pixel')
        self.deformable_attention = DeformableAttention(
            dim = 256,                      # feature dimensions
            dim_head = 16,                  # dimension per head
            heads = 16,                     # attention heads
            dropout = 0.1,                  # dropout
            downsample_factor = 4,          # downsample factor (r in paper)
            offset_scale = 4,               # scale of offset, maximum offset
            offset_groups = 1,              # number of offset groups, a factor of heads
            offset_kernel_size = 6,         # offset kernel size
        )
        self.deformable_attention.forward = partial(_forward, self.deformable_attention)

        # 1D Convolution layer used to rescale the permuted output of deformable attention to fit TSA output
        # 32 x 312 x 256 -> 32 x 1 x 256
        self.conv = nn.Conv1d(12*26, 1, 1)

        # LayerNorm and Dropout for better generalization and smoother gradients
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, q=None):

        # DAT and 1x1 Convolution to use query to attend to a single image feature map, resulting in
        # a image embedding of the same shape as query
        output = self.deformable_attention(x, q)
        bs = x.shape[0]
        output = output.view(bs, 256, -1).permute(0, 2, 1)
        output = self.conv(output)

        # LayerNorm, Dropout, and Skip Connections
        output = self.norm(self.dropout(output) + q)

        return output
