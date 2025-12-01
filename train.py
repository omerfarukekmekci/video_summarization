import argparse
from pathlib import Path

import torch
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--topk", type=int, default=5, help="Number of frames for diversity term"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Pad/truncate sequences to this length; also used for positional encoding",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)

    dataset = SumMeTVSumDataset(
        h5_path=args.dataset,
        split=args.split,
        split_file=args.split_file,
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
    loss_fn = VideoSummaryLoss(target_ratio=0.15)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            features = batch["features"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            target_scores = batch["target_scores"].to(device)

            tokens = model.frame_encoder(features, frame_mask=frame_mask)
            encoded = model.transformer(tokens, mask=frame_mask)
            scores = model.head(encoded)

            selected_feats = None
            if args.topk > 0:
                selected_feats, _ = hard_topk_frames(
                    encoded, scores, mask=frame_mask, k=args.topk
                )

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
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()
