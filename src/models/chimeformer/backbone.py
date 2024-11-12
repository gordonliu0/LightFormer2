from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from models.lightformer.identity import Identity

class Backbone(nn.Module):
    def __init__(self, config):
        """
        Resnet-18 with last two layers (fc and avg pooling) replaced with convolutions and batch norms.
        """
        super().__init__()

        self.embed_dim = config["model"]["embedding_dim"]

        # Remove last two layers: Average Pooling and Fully Connected
        self.resnet = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.layer4 = Identity()
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()

        # Add an additional downsampling step through conv relu batch sequences
        self.down_conv = nn.Sequential(
            nn.Conv2d(256, self.embed_dim, 3),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 5),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 7),
            nn.GELU(),
            nn.BatchNorm2d(config["model"]["embedding_dim"]),
        )

    def forward(self, x):
        """
        x: image buffer tensor of size batch_size, image_count, channels, height, width
        Returns an image feature map tensor of size batch_size, image_count, 256 channels, height-4, width-4
        """
        batch_size, image_count, channels, height, width = x.shape
        x = x.reshape(batch_size * image_count, channels, height, width)
        x = self.resnet(x)
        x = x.view(batch_size*image_count, 256, 32, 60) # Re
        x = self.down_conv(x)
        _, c, h, w = x.shape
        x = x.reshape(batch_size, image_count, c, h, w)
        return x
