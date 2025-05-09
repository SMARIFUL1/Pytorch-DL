import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .encoder_block import TransformerEncoderBlock

class BanknoteTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=8, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embed)
        x = x.transpose(0, 1)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[0]
        return self.head(cls_out)
