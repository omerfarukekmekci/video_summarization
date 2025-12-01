import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from datasets import SumMeTVSumDataset, video_summary_collate
from transformer_encoder import VideoSummarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate video summarizer on SumMe/TVSum using precomputed HDF5 features"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to ECCV16 .h5 file",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="JSON split file describing train/val/test ids",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name to evaluate (e.g. 'val' or 'test')",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a model checkpoint saved with torch.save(model.state_dict(), ...)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size (use 1 if you rely on per-video metadata)",
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Pad/truncate sequences to this length; must match training",
    )
    parser.add_argument(
        "--length-ratio",
        type=float,
        default=0.15,
        help="Target summary length as a fraction of the sequence (e.g. 0.15 = 15%%)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    return parser.parse_args()


def build_model(
    dataset: SumMeTVSumDataset, args: argparse.Namespace
) -> VideoSummarizer:
    if not hasattr(dataset, "feature_dim"):
        raise AttributeError(
            "Dataset is missing 'feature_dim'. Ensure SumMeTVSumDataset exposes it "
            "and that you are using the precomputed ECCV16 HDF5 features."
        )

    model = VideoSummarizer(
        feature_dim=dataset.feature_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. "
            "Make sure you have trained and saved a model first."
        )

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the current model definition.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}\n"
            "Verify that d_model, num_heads, num_layers and feature_dim match "
            "those used during training."
        )

    return model


def build_machine_summary(
    scores: torch.Tensor, frame_mask: torch.BoolTensor, length_ratio: float
) -> torch.Tensor:
    """
    Builds a simple frame-level summary by selecting top-k scored frames under
    a length budget defined by `length_ratio`.

    Both `scores` and the returned summary are 1D tensors over the (unpadded)
    temporal dimension used during training.
    """

    if scores.ndim != 1 or frame_mask.ndim != 1:
        raise ValueError(
            "Expected 1D tensors for scores and frame_mask after squeezing, "
            f"but received shapes scores={tuple(scores.shape)}, "
            f"frame_mask={tuple(frame_mask.shape)}."
        )

    valid_idx = frame_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return torch.zeros_like(scores, dtype=torch.float32)

    valid_scores = scores[valid_idx]
    max_len = valid_scores.numel()

    if not (0.0 < length_ratio <= 1.0):
        raise ValueError(
            f"length_ratio must be in (0, 1], but got {length_ratio}. "
            "Typical values for summarization are around 0.15."
        )

    k = max(1, int(length_ratio * max_len))
    k = min(k, max_len)

    topk_indices = torch.topk(valid_scores, k=k, dim=0).indices
    summary = torch.zeros_like(scores, dtype=torch.float32)
    summary[valid_idx[topk_indices]] = 1.0
    return summary


def f_score(
    machine_summary: torch.Tensor, user_summaries: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Computes precision, recall and F-score between a machine summary and
    multiple user summaries.

    machine_summary: (T,)
    user_summaries: (num_users, T)
    """

    if machine_summary.ndim != 1:
        raise ValueError(
            f"machine_summary must be 1D (T,), but has shape {tuple(machine_summary.shape)}"
        )

    if user_summaries.ndim != 2:
        raise ValueError(
            f"user_summaries must have shape (num_users, T), "
            f"but has shape {tuple(user_summaries.shape)}"
        )

    machine = machine_summary.bool().unsqueeze(0)  # (1, T)
    users = user_summaries.bool()  # (num_users, T)

    intersection = (machine & users).sum(dim=1).float()
    machine_sum = machine.sum(dim=1).float().clamp(min=1e-6)
    user_sum = users.sum(dim=1).float().clamp(min=1e-6)

    precision = (intersection / machine_sum).mean().item()
    recall = (intersection / user_sum).mean().item()

    if precision + recall == 0:
        f = 0.0
    else:
        f = 2 * precision * recall / (precision + recall)

    return precision, recall, f


def evaluate() -> None:
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
        shuffle=False,
        num_workers=4,
        collate_fn=video_summary_collate,
        pin_memory=True,
    )

    model = build_model(dataset, args).to(device)
    model.eval()

    all_f = []
    all_prec = []
    all_rec = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)  # (B, T, feat_dim)
            frame_mask = batch["frame_mask"].to(device)  # (B, T)
            user_summary = batch["user_summary"].to(device)  # (B, num_users, T)

            scores = model(features, frame_mask=frame_mask)  # (B, T)

            # Evaluate per video in the batch
            for i in range(scores.size(0)):
                vid_scores = scores[i]  # (T,)
                vid_mask = frame_mask[i]  # (T,)
                vid_users = user_summary[i]  # (num_users, T)

                # Trim to valid region based on mask
                valid_len = int(vid_mask.sum().item())
                if valid_len == 0:
                    continue

                vid_scores = vid_scores[:valid_len]
                vid_mask_valid = vid_mask[:valid_len]
                vid_users_valid = vid_users[:, :valid_len]

                machine = build_machine_summary(
                    vid_scores, vid_mask_valid, length_ratio=args.length_ratio
                )
                prec, rec, f = f_score(machine, vid_users_valid)

                all_prec.append(prec)
                all_rec.append(rec)
                all_f.append(f)

    if not all_f:
        raise RuntimeError(
            "No valid videos were evaluated. This may happen if all frame masks are "
            "empty; check your dataset and max_seq_len settings."
        )

    mean_prec = sum(all_prec) / len(all_prec)
    mean_rec = sum(all_rec) / len(all_rec)
    mean_f = sum(all_f) / len(all_f)

    print(
        f"Evaluation on split='{args.split}': "
        f"Precision={mean_prec:.4f}, Recall={mean_rec:.4f}, F-score={mean_f:.4f}"
    )


if __name__ == "__main__":
    evaluate()
