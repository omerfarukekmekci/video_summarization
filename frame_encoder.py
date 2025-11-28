import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class FrameEncoder(nn.Module):
    def __init__(self, d_model=512, max_seq_len=2048, apply_transforms=True):
        super().__init__()

        # pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # removing the classifier, but we keep the feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # output shape = (batch, 512, 1, 1)

        # sets the dimension to the desired value that is d_model
        self.proj = nn.Linear(512, d_model)

        self.apply_transforms = apply_transforms
        if self.apply_transforms:
            # torchvision transforms applied per-frame
            self.transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        # positional encoding buffer (1, max_seq_len, d_model)
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

    def forward(self, frames):
        b, t, c, h, w = frames.shape
        device = frames.device

        # flatten batch and time to convert to the dimensions resnet expects
        x = frames.reshape(b * t, c, h, w)

        if self.apply_transforms:
            processed = []
            for frame in x:
                processed.append(self.transform(frame.cpu()))
            x = torch.stack(processed, dim=0).to(device)

        # CNN features
        feats = self.backbone(x)
        feats = feats.squeeze(-1).squeeze(-1)  # from (b*t, 512, 1, 1) to (b*t, 512)

        feats = self.proj(feats)  # (b*t, d_model)

        # Unflatten back to (batch, seq_len, d_model)
        feats = feats.reshape(b, t, -1)

        if t > self.positional_encoding.size(1):
            raise ValueError(
                f"Sequence length {t} exceeds maximum {self.positional_encoding.size(1)}"
            )

        # add positional encoding so the model knows the order of the frames
        pos_enc = self.positional_encoding[:, :t, :].to(device)
        feats = feats + pos_enc

        return feats
