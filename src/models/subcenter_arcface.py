import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class SubcenterArcface(nn.Module):
    """
    Implementation of subcenter large margin arc distance.
    """
    def __init__(self, in_features, out_features, K, s=64.0, m=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.cluster_centres = K
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(self.out_features, self.cluster_centres, self.in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):

        # ( out_features x cluster_centres x in_features ) * ( B x 1 x 1 x in_features ) = ( B x out_features x cluster_centres x in_features )
        # Then sum to get ( B x out_features x cluster_centres )
        B, in_features = x.shape
        W_norm = F.normalize(self.W, dim=2)
        x_norm = F.normalize(x, dim=1).view(B, 1, 1, in_features)
        element_wise = W_norm * x_norm
        subclass_cosine_sim = element_wise.sum(dim=3)

        # next, convert to angles, add margin, convert back, and scale to help gradients
        class_cosine_sim = subclass_cosine_sim.max(dim=2).values
        theta = class_cosine_sim.arccos()
        arcface = self.s * (theta + self.m).cos()

        return arcface
