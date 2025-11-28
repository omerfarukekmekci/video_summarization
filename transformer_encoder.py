import torch
import torch.nn as nn
from frame_encoder import FrameEncoder
from transformer import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, num_layers=3, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class FrameScoringHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.scorer(x).squeeze(-1)  # (batch, seq_len)


class VideoSummarizer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=3):
        super().__init__()
        self.frame_encoder = FrameEncoder(d_model)
        self.transformer = TransformerEncoder(d_model, num_layers, num_heads)
        self.head = FrameScoringHead(d_model)

    def forward(self, frames, frame_mask=None):
        x = self.frame_encoder(frames)
        scores = self.head(self.transformer(x, mask=frame_mask))
        return scores
