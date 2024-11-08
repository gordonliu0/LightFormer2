from functools import partial
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from types import MethodType
from models.lightformer.identity import Identity

class Backbone(nn.Module):
    def __init__(self):
        """
        Resnet-18 with last two layers (fc and avg pooling) replaced with convolutions and batch norms.
        """
        super().__init__()

        # Remove last two layers: Average Pooling and Fully Connected
        self.resnet = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()

        # Add a downsampling step through conv relu batch sequences
        self.down_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        """
        x: image buffer tensor of size batch_size, image_count, channels, height, width
        Returns an image feature map tensor of size batch_size, image_count, 256 channels, height-4, width-4
        """
        batch_size, image_count, channels, height, width = x.shape
        x = x.reshape(batch_size * image_count, channels, height, width)
        x = self.resnet(x)
        x = x.view(batch_size*image_count, 512, 16, 30)
        x = self.down_conv(x)
        _, c, h, w = x.shape
        x = x.reshape(batch_size, image_count, c, h, w)
        return x
