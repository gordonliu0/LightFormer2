import torch
import torch.nn as nn

def get_reference_points(H=4, W=4, bs=8, device='mps'):
    ref_y, ref_x = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float, device=device), torch.linspace(-1, 1, W, dtype=torch.float, device=device))
    ref_y = ref_y.reshape(-1)[None]
    ref_x = ref_x.reshape(-1)[None]
    ref_2d = torch.stack((ref_x, ref_y), -1)
    # ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    return ref_2d

print(get_reference_points())
