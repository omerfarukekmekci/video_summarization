import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, original, reconstructed):
        if original is None or reconstructed is None:
            return torch.zeros(
                (),
                device=(
                    original.device if original is not None else reconstructed.device
                ),
            )
        return self.mse(reconstructed, original)


class SparsityLoss(nn.Module):
    def __init__(self, target_ratio=0.15):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, scores, mask=None):
        if scores is None:
            raise ValueError("scores tensor is required for sparsity loss")

        if mask is not None:
            valid_scores = scores.masked_select(mask)
        else:
            valid_scores = scores.reshape(-1)

        if valid_scores.numel() == 0:
            return torch.zeros((), device=scores.device)

        return (valid_scores.mean() - self.target_ratio) ** 2


def diversity_loss(feats):
    if feats is None or feats.numel() == 0:
        return torch.tensor(0.0, device=feats.device if feats is not None else "cpu")

    if feats.shape[1] < 2:
        return torch.zeros((), device=feats.device)

    norm = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    sim = norm @ norm.transpose(1, 2)
    b, k, _ = sim.shape
    sim = sim - torch.eye(k, device=sim.device).unsqueeze(0)
    return sim.mean()


class VideoSummaryLoss(nn.Module):
    def __init__(
        self,
        target_ratio=0.15,
        recon_weight=0.0,
        sparsity_weight=3.0,
        diversity_weight=0.0,
        regression_weight=1.0,
    ):
        super().__init__()
        self.reconstruction = ReconstructionLoss()
        self.sparsity = SparsityLoss(target_ratio)
        self.recon_weight = recon_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.regression_weight = regression_weight

    def forward(
        self,
        frames=None,
        recon=None,
        scores=None,
        selected_feats=None,
        target_scores=None,
        frame_mask=None,
    ):
        total = torch.tensor(
            0.0,
            device=(
                scores.device
                if scores is not None
                else (recon.device if recon is not None else torch.device("cpu"))
            ),
        )

        if self.recon_weight > 0 and frames is not None and recon is not None:
            total = total + self.recon_weight * self.reconstruction(frames, recon)

        if (
            self.regression_weight > 0
            and target_scores is not None
            and scores is not None
        ):
            pred = scores
            target = target_scores
            if frame_mask is not None:
                pred = pred.masked_select(frame_mask)
                target = target.masked_select(frame_mask)
            else:
                pred = pred.reshape(-1)
                target = target.reshape(-1)

            # Train with raw scores (no sigmoid)
            # Normalize target to [0, 1] range
            target_norm = torch.clamp(target, 0.0, 1.0)
            total = total + self.regression_weight * F.mse_loss(pred, target_norm)

        if self.sparsity_weight > 0 and scores is not None:
            # Apply sparsity directly to raw scores
            total = total + self.sparsity_weight * self.sparsity(scores, frame_mask)

        if self.diversity_weight > 0 and selected_feats is not None:
            total = total + self.diversity_weight * diversity_loss(selected_feats)

        return total
