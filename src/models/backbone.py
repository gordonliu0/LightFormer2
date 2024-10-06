from functools import partial
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def _resnet_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x) # C: 256
    x = self.layer4(x) # C: 512
    return x

class Backbone(nn.Module):
    def __init__(self):
        """
        Resnet-18 with last two layers (fc and avg pooling) replaced with convolutions and batch norms.
        """
        super().__init__()

        # Remove last two layers: Average Pooling and Fully Connected
        self.resnet = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = None
        self.resnet.avgpool = None

        # Adjust resnet forward to reflect removed last two layers
        self.resnet.forward = partial(_resnet_forward, self.resnet)

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
        print(x.shape)
        x = x.reshape(batch_size * image_count, channels, height, width)
        x = self.resnet(x)
        x = self.down_conv(x)
        _, c, h, w = x.shape
        print(x.shape)
        x = x.reshape(batch_size, image_count, c, h, w)
        return x
