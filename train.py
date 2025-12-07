import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import SumMeTVSumDataset, video_summary_collate
from frame_selection import hard_topk_frames
from losses import VideoSummaryLoss
from transformer_encoder import VideoSummarizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train video summarizer on SumMe/TVSum"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to ECCV16 .h5 file"
    )
    parser.add_argument("--split-file", type=Path, default=None, help="JSON split file")
    parser.add_argument(
        "--split", type=str, default=None, help="train/val/test split name"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to save model.state_dict() after training. "
            "If omitted, no checkpoint is written."
        ),
    )
    parser.add_argument(
        "--use-precomputed",
        action="store_true",
        help=(
            "Deprecated flag kept for backward compatibility. "
            "Precomputed ECCV16 features are always used in this version."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--topk", type=int, default=0, help="Number of frames for diversity term"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Pad/truncate sequences to this length; also used for positional encoding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For cross-validation: if a split is specified, use all videos EXCEPT that split for training
    train_video_ids = None
    if args.split_file is not None and args.split is not None:
        # Load all video IDs from the dataset
        with h5py.File(args.dataset, "r") as h5_file:
            all_video_ids = sorted(h5_file.keys())

        # Load the test split IDs
        split_data = json.loads(Path(args.split_file).read_text())
        if args.split not in split_data:
            raise ValueError(f"Split '{args.split}' not found in {args.split_file}")
        test_video_ids = set(split_data[args.split])

        # Exclude test split videos from training
        train_video_ids = [vid for vid in all_video_ids if vid not in test_video_ids]

        if len(train_video_ids) == 0:
            raise ValueError(
                f"No training videos found after excluding test split '{args.split}'. "
                "This suggests all videos are in the test split, which is invalid."
            )

        print(
            f"Training on {len(train_video_ids)} videos (excluding {len(test_video_ids)} test videos from split '{args.split}')"
        )

    dataset = SumMeTVSumDataset(
        h5_path=args.dataset,
        split=None,  # Don't use split parameter when we have explicit video_ids
        split_file=None,
        video_ids=train_video_ids,  # Use explicit list of training videos
        max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=video_summary_collate,
        pin_memory=True,
    )

    if not hasattr(dataset, "feature_dim"):
        raise AttributeError(
            "Dataset is missing 'feature_dim'. Ensure SumMeTVSumDataset exposes it."
        )

    model = VideoSummarizer(
        feature_dim=dataset.feature_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Use learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    # Slightly down-weight sparsity and diversity so the model can focus more
    # on matching the human importance scores.
    loss_fn = VideoSummaryLoss(
        target_ratio=0.15,
        sparsity_weight=3.0,
        diversity_weight=0.0,
        regression_weight=1.0,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_reg_mse = 0.0
        running_sparsity = 0.0
        running_corr = 0.0
        num_batches = 0

        for batch in dataloader:
            features = batch["features"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            target_scores = batch["target_scores"].to(device)

            tokens = model.frame_encoder(features, frame_mask=frame_mask)
            encoded = model.transformer(tokens, mask=frame_mask)
            scores = model.head(encoded)

            selected_feats = None

            loss = loss_fn(
                frames=None,
                recon=None,
                scores=scores,
                selected_feats=selected_feats,
                target_scores=target_scores,
                frame_mask=frame_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # --- Metrics for monitoring (no-grad to avoid graph retention) ---
            with torch.no_grad():
                # Regression MSE on valid frames (using normalized scores for monitoring)
                pred = scores
                target = target_scores
                if frame_mask is not None:
                    valid_mask = frame_mask
                    pred_flat = pred[valid_mask]
                    target_flat = target[valid_mask]
                else:
                    pred_flat = pred.reshape(-1)
                    target_flat = target.reshape(-1)

                if pred_flat.numel() > 0:
                    # Use normalized scores for monitoring (matching loss function)
                    pred_norm = torch.sigmoid(pred_flat)
                    target_norm = torch.clamp(target_flat, 0.0, 1.0)
                    reg_mse = F.mse_loss(pred_norm, target_norm)
                else:
                    reg_mse = torch.tensor(0.0, device=device)

                # Sparsity term (same as inside VideoSummaryLoss, but unweighted)
                probs = torch.sigmoid(scores)
                if frame_mask is not None:
                    valid_probs = probs[frame_mask]
                else:
                    valid_probs = probs.reshape(-1)

                if valid_probs.numel() > 0:
                    sparsity = (valid_probs.mean() - loss_fn.sparsity.target_ratio) ** 2
                else:
                    sparsity = torch.tensor(0.0, device=device)

                # Correlation between prediction and target (how well shapes match)
                if pred_flat.numel() > 1:
                    p_center = pred_flat - pred_flat.mean()
                    t_center = target_flat - target_flat.mean()
                    denom = (p_center.norm() * t_center.norm()).clamp_min(1e-8)
                    corr = (p_center * t_center).sum() / denom
                else:
                    corr = torch.tensor(0.0, device=device)

                running_reg_mse += reg_mse.item()
                running_sparsity += sparsity.item()
                running_corr += corr.item()

        if num_batches == 0:
            raise RuntimeError("No batches were processed during training epoch.")

        avg_loss = running_loss / num_batches
        avg_reg_mse = running_reg_mse / num_batches
        avg_sparsity = running_sparsity / num_batches
        avg_corr = running_corr / num_batches

        # Update learning rate scheduler
        scheduler.step()

        # Print summary every epoch; you can watch trends and especially every 10th.
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"loss: {avg_loss:.4f}, "
            f"reg_mse: {avg_reg_mse:.4f}, "
            f"sparsity: {avg_sparsity:.4f}, "
            f"corr: {avg_corr:.4f}, "
            f"lr: {current_lr:.6f}"
        )

    # Save checkpoint at the end of training if a path is provided.
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
