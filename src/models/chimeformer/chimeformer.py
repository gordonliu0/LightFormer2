from torch import nn
from models.chimeformer.backbone import Backbone
from models.chimeformer.encoder import Encoder
from models.chimeformer.decoder import Decoder

class ChimeFormer(nn.Module):
    def __init__(self, config):
        """
        Two headed classification architecture, featuring Encoder-only architecture based on SOTA Vision Transformer models, with embedding network (pretrained on driving tasks) outputting frame-level embeddings for temporal local bias,
        """
        super().__init__()
        self.backbone = Backbone(config=config)
        self.query_embed = nn.Embedding(1, config["model"]["embedding_dim"])
        self.encoder = Encoder(config=config)
        self.mlp = nn.Sequential(
            nn.Linear(config["model"]["embedding_dim"], 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
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
        # Shared upsampling MLP
        x = self.mlp(x)
        x = x.squeeze(1)
        # Dual Class Decoders
        st_class = self.decoder_1(x)
        lf_class = self.decoder_2(x)
        return st_class, lf_class
