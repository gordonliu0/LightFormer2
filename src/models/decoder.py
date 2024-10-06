import torch.nn as nn
import torch.nn.functional as F
from models.subcenter_arcface import SubcenterArcface

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.K = 4
        self.out_class_num = 2

        # K = 3 used in Sub-Center ArcFace 2020, w=20 used in LightFormer 2023
        # s = 64.0 in ArcFace 2018, Sub-Center ArcFace 2020
        # m = 0.5 in both LightFormer 2023, Sub-Center ArcFace 2020
        self.sub_arcface = SubcenterArcface(
            in_features = 1024,
            out_features = self.out_class_num,
            K = self.K,
            s = 64.0,
            m = 0.5,
        )

    def forward(self, x):
        arcface = self.sub_arcface(x)
        res = F.softmax(arcface, dim=1)
        return res
