import torch.nn as nn
import torch.nn.functional as F
from models.lightformer.subcenter_arcface import SubcenterArcface

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        return self.linear(x)
