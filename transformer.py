import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=None, dropout=0.05):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # normalization layers to use llater
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # feed-forward network, applied to each position independently
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Dropout layer used after attention and ff
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # normalize before attention to stabilize training
        x_norm = self.norm1(x)

        attn_out = self.self_attn(x_norm, mask=mask)

        attn_out = self.dropout(attn_out)
        x = x + attn_out  # residual connection

        # Feed-forward layer
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)  # shape: (b, t, d_model)
        ff_out = self.dropout(ff_out)
        x = x + ff_out

        return x
