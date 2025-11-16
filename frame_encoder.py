import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class FrameEncoder(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        # pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # removing the classifier, but we keep the feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # output shape = (batch, 512, 1, 1)

        # sets the dimension to the desired value that is d_model
        self.proj = nn.Linear(512, d_model)

        # transforms for frames
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, frames):
        b, t, c, h, w = frames.shape
        # (batch size, frame count, channels, height, width)

        # flatten batch and time to convert to the dimensions resnet expects
        x = frames.reshape(b * t, c, h, w)

        # CNN features
        feats = self.backbone(x)
        feats = feats.squeeze(-1).squeeze(-1)  # from (b*t, 512, 1, 1) to (b*t, 512)

        feats = self.proj(feats)  # (b*t, d_model)

        # Unflatten back to (batch, seq_len, d_model)
        feats = feats.reshape(b, t, -1)

        feats = feats + self.positional_encoding[:, :T, :]

        return feats
