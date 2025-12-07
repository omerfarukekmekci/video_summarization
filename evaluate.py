import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
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
        required=False,
        help=(
            "Path to a model checkpoint saved with torch.save(model.state_dict(), ...). "
            "If omitted together with --oracle, evaluation will use oracle scores instead "
            "of a trained model."
        ),
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
    parser.add_argument(
        "--oracle",
        action="store_true",
        help=(
            "If set, bypass the model and use ground-truth target scores to build "
            "summaries. This is useful to sanity-check the evaluation pipeline and "
            "establish an approximate upper bound on achievable F-score."
        ),
    )
    parser.add_argument(
        "--eval-method",
        type=str,
        default="max",
        choices=["max", "avg"],
        help="How to aggregate F-scores across users: 'max' or 'avg'. "
        "This mirrors the ECCV16 evaluation code.",
    )
    return parser.parse_args()


def build_model(
    dataset: SumMeTVSumDataset, args: argparse.Namespace
) -> VideoSummarizer:
    if args.oracle:
        raise ValueError(
            "build_model should not be called when --oracle is set. "
            "Disable --oracle or omit --checkpoint to use a trained model."
        )

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

    if args.checkpoint is None or not args.checkpoint.exists():
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


def knapsack(capacity: int, lengths: List[int], scores: List[float]) -> List[int]:
    """
    0/1 knapsack that mirrors the behaviour expected by generate_summary.py:
    returns the indices of selected segments.
    """

    n = len(lengths)
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = lengths[i - 1]
        v = scores[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                cand = dp[i - 1][c - w] + v
                if cand > dp[i][c]:
                    dp[i][c] = cand

    # Backtrack to recover selected indices
    res = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            res.append(i - 1)
            c -= lengths[i - 1]
            if c <= 0:
                break
    res.reverse()
    return res


def generate_summary_single(
    shot_bound: np.ndarray,
    frame_init_scores: np.ndarray,
    n_frames: int,
    positions: np.ndarray,
    length_ratio: float,
) -> np.ndarray:
    """
    Per-video version of UnsupervisedVideoSummarization's generate_summary.py.
    """

    # Compute the importance scores for the initial frame sequence
    frame_scores = np.zeros((int(n_frames)), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [int(n_frames)]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(frame_init_scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = frame_init_scores[i]

    # Shot-level importance scores
    shot_imp_scores: List[float] = []
    shot_lengths: List[int] = []
    for shot in shot_bound:
        start, end = int(shot[0]), int(shot[1])
        length = end - start + 1
        shot_lengths.append(length)
        shot_imp_scores.append(float(frame_scores[start : end + 1].mean()))

    # Select best shots using knapsack
    total_len = int(shot_bound[-1][1] + 1)
    final_max_length = int(total_len * float(length_ratio))
    final_max_length = max(1, final_max_length)

    selected = knapsack(final_max_length, shot_lengths, shot_imp_scores)

    # Build binary summary
    summary = np.zeros(total_len, dtype=np.int8)
    for shot_idx in selected:
        start, end = shot_bound[shot_idx]
        start, end = int(start), int(end)
        summary[start : end + 1] = 1

    return summary


def evaluate_summary_numpy(
    predicted_summary: np.ndarray, user_summary: np.ndarray, eval_method: str
) -> float:
    """
    Direct port of evaluation_metrics.evaluate_summary.
    Returns F-score in [0, 100].
    """

    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[: len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[: user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        # Compute precision, recall, f-score
        if S.sum() == 0 or G.sum() == 0:
            f_scores.append(0.0)
            continue

        precision = overlapped.sum() / S.sum()
        recall = overlapped.sum() / G.sum()
        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append(2 * precision * recall * 100.0 / (precision + recall))

    if not f_scores:
        return 0.0

    if eval_method == "max":
        return float(max(f_scores))
    else:
        return float(sum(f_scores) / len(f_scores))


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

    model = None
    if not args.oracle:
        if args.checkpoint is None:
            raise ValueError(
                "A checkpoint must be provided unless --oracle is set. "
                "Either pass --checkpoint path or use --oracle to evaluate "
                "using ground-truth scores."
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
            target_scores = batch["target_scores"].to(device)  # (B, T)
            change_points_batch = batch.get("change_points", None)
            picks_batch = batch.get("picks", None)
            n_frames_batch = batch.get("n_frames", None)

            if args.oracle:
                # Use ground-truth target scores (e.g. gtscore) as ideal importance
                # scores. This allows us to measure an approximate upper bound given
                # the current evaluation protocol.
                scores = target_scores
            else:
                scores = model(features, frame_mask=frame_mask)  # (B, T)

            # Evaluate per video in the batch
            for i in range(scores.size(0)):
                vid_scores = scores[i]  # (T,)
                vid_mask = frame_mask[i]  # (T,)
                vid_users = user_summary[i]  # (num_users, T)
                vid_change_points = None
                vid_picks = None
                vid_n_frames = None
                if change_points_batch is not None:
                    vid_change_points = change_points_batch[i]
                if picks_batch is not None:
                    vid_picks = picks_batch[i]
                if n_frames_batch is not None:
                    vid_n_frames = int(n_frames_batch[i].item())

                # Trim to valid region based on mask
                valid_len = int(vid_mask.sum().item())
                if valid_len == 0:
                    continue

                vid_scores = vid_scores[:valid_len]
                vid_mask_valid = vid_mask[:valid_len]
                vid_users_valid = vid_users[:, :valid_len]

                if (
                    vid_change_points is None
                    or vid_picks is None
                    or vid_n_frames is None
                ):
                    raise RuntimeError(
                        "Missing change_points, picks or n_frames for ECCV16 evaluation. "
                        "Ensure datasets.py is returning these fields correctly."
                    )

                # Convert to numpy for generate_summary / evaluate_summary.
                scores_tensor = vid_scores.cpu()
                # Normalize to [0, 1] using min-max scaling per video
                if scores_tensor.numel() > 0:
                    s_min = scores_tensor.min()
                    s_max = scores_tensor.max()
                    if s_max > s_min:
                        scores_normalized = (scores_tensor - s_min) / (s_max - s_min)
                    else:
                        scores_normalized = torch.ones_like(scores_tensor) * 0.5
                else:
                    scores_normalized = scores_tensor
                scores_np = scores_normalized.numpy().astype(np.float32)

                # change_points and picks are small; handle missing .device carefully.
                cp_np = vid_change_points.cpu().numpy()
                picks_np = vid_picks.cpu().numpy()
                user_np = vid_users_valid.cpu().numpy()

                summary_np = generate_summary_single(
                    shot_bound=cp_np,
                    frame_init_scores=scores_np,
                    n_frames=vid_n_frames,
                    positions=picks_np,
                    length_ratio=args.length_ratio,
                )
                f = evaluate_summary_numpy(summary_np, user_np, args.eval_method)

                all_f.append(f)

    if not all_f:
        raise RuntimeError(
            "No valid videos were evaluated. This may happen if all frame masks are "
            "empty; check your dataset and max_seq_len settings."
        )

    mean_f = sum(all_f) / len(all_f)

    print(f"Evaluation on split='{args.split}': F-score={mean_f:.4f}")


if __name__ == "__main__":
    evaluate()
