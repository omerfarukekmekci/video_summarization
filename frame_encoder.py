import math
import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """
    Lightweight token encoder that projects precomputed ECCV16 features into the
    transformer dimension and injects sinusoidal positional encodings.
    """

    def __init__(self, feature_dim, d_model=512, max_seq_len=4096, dropout=0.05):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.proj = nn.Linear(feature_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        pos_encoding = self._build_positional_encoding(max_seq_len, d_model)
        self.register_buffer("positional_encoding", pos_encoding, persistent=False)

    def _build_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, features, frame_mask=None):
        if features.dim() != 3:
            raise ValueError(
                "FrameEncoder expects features shaped (batch, seq_len, feature_dim) "
                f"but received tensor with shape {tuple(features.shape)}."
            )

        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim} but received "
                f"{features.size(-1)}. Ensure you are using the ECCV16 HDF5 features."
            )

        b, t, _ = features.shape
        if t > self.positional_encoding.size(1):
            raise ValueError(
                f"Sequence length {t} exceeds FrameEncoder max_seq_len "
                f"{self.positional_encoding.size(1)}. Increase max_seq_len if needed."
            )

        x = self.proj(features)
        pos_enc = self.positional_encoding[:, :t, :].to(x.device)
        x = self.dropout(x + pos_enc)

        if frame_mask is not None:
            if frame_mask.dim() != 2:
                raise ValueError(
                    "frame_mask must have shape (batch, seq_len) to match features, "
                    f"but received {tuple(frame_mask.shape)}."
                )
            mask = frame_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * mask

        return x
