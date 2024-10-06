from torch import nn
from models.backbone import Backbone
from models.encoder import Encoder
from models.decoder import Decoder

class LightFormer(nn.Module):
    def __init__(self):
        """
        Ming et al 2023: Modified transformer architecture featuring image buffer, encoder, two decoders, image embedding query, temporal self attention, and multi arcface loss.
        """
        super().__init__()
        self.backbone = Backbone()
        self.query_embed = nn.Embedding(1, 256)
        self.encoder = Encoder()
        self.mlp = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
        )
        self.head_1 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
        )
        self.head_2 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
        )
        self.decoder_1 = Decoder()
        self.decoder_2 = Decoder()

    def forward(self, x):
        """
        x: image buffer of size 10
        """
        # Backbone
        x = self.backbone(x)
        # Encoder
        query = self.query_embed.weight
        x = self.encoder(query, x)
        # MLP
        x = self.mlp(x)
        # Two-headed MLP
        x_h1, x_h2 = self.head_1(x), self.head_2(x)
        x_h1, x_h2 = x_h1.unsqueeze(3)
        # Dual Class Decoders
        st_class = self.decoder_1(x_h1)
        lf_class = self.decoder_2(x_h2)
        return st_class, lf_class
