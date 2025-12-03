import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import torch
from torch.utils.data import Dataset


@dataclass
class VideoSummaryItem:
    """
    Container returned by SumMe/TVSum datasets.

    Attributes
    ----------
    video_id: str
        Name of the video inside the HDF5 file (e.g. "video_1").
    features: torch.Tensor
        Feature sequence of shape (seq_len, feature_dim). These are typically
        pool5 features that you can either feed directly to a lightweight head
        or re-project through the existing FrameEncoder if you would like the
        ResNet backbone to be fine-tuned.
    frame_mask: torch.BoolTensor
        Mask with True for valid timesteps and False for padding.
    target_scores: torch.Tensor
        Aggregated per-frame importance scores computed from the user
        annotations. This is what you can use as a regression target.
    user_summary: torch.Tensor
        Raw binary summaries provided by annotators, shape (num_users, seq_len).
        These are necessary for computing the dataset F-scores at evaluation.
    change_points: Optional[torch.Tensor]
        Change point boundaries (segments) from the dataset metadata.
    picks: Optional[torch.Tensor]
        Frame indices that correspond to `features`.
    n_frames: int
        Total number of frames in the original video before uniform sampling.
    """

    video_id: str
    features: torch.Tensor
    frame_mask: torch.BoolTensor
    target_scores: torch.Tensor
    user_summary: torch.Tensor
    change_points: Optional[torch.Tensor]
    picks: Optional[torch.Tensor]
    n_frames: int


def load_split_mapping(
    split_file: Optional[Path], split: Optional[str]
) -> Optional[Sequence[str]]:
    """
    Loads a list of video ids for a given split from a JSON file.

    Expected JSON schema:
    {
        "train": ["video_1", "video_3", ...],
        "val": ["video_4", ...],
        "test": ["video_2", ...]
    }
    """

    if split_file is None or split is None:
        return None

    data = json.loads(Path(split_file).read_text())
    if split not in data:
        raise ValueError(f"Split '{split}' not found in {split_file}")
    return data[split]


class SumMeTVSumDataset(Dataset):
    """
    Accesses the official ECCV16 HDF5 releases for SumMe or TVSum.

    Parameters
    ----------
    h5_path : str or Path
        Path to the dataset HDF5 file (e.g. eccv16_dataset_summe_google_pool5.h5).
    split : Optional[str]
        Named split to use ("train", "val", "test"). Ignored if `video_ids` is provided.
    split_file : Optional[str or Path]
        JSON file describing the video ids per split. Required if `split` is set.
    video_ids : Optional[Sequence[str]]
        Explicit list of video ids to use instead of reading from the split file.
    aggregate : str
        How to collapse user annotations into a single regression target.
        Choices: {"mean", "median", "max"}.
    max_seq_len : Optional[int]
        If provided, feature sequences are padded or truncated to this length.
    pad_value : float
        Value used when padding features.
    """

    def __init__(
        self,
        h5_path: Path,
        split: Optional[str] = None,
        split_file: Optional[Path] = None,
        video_ids: Optional[Sequence[str]] = None,
        aggregate: str = "mean",
        max_seq_len: Optional[int] = None,
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        allowed_ids = video_ids or load_split_mapping(split_file, split)
        with h5py.File(self.h5_path, "r") as h5_file:
            all_ids = sorted(h5_file.keys())

        if allowed_ids is not None:
            all_ids = [vid for vid in all_ids if vid in set(allowed_ids)]
            missing = set(allowed_ids) - set(all_ids)
            if missing:
                raise ValueError(
                    f"Video ids not present in {self.h5_path}: {sorted(missing)}"
                )

        if not all_ids:
            raise ValueError(
                f"No videos found in HDF5 file {self.h5_path}. "
                "Verify the file matches the ECCV16 release."
            )

        self.video_ids: List[str] = all_ids
        self.aggregate = aggregate
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.feature_dim = self._infer_feature_dim()

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> VideoSummaryItem:
        video_id = self.video_ids[index]
        with h5py.File(self.h5_path, "r") as h5_file:
            group = h5_file[video_id]
            features = torch.from_numpy(group["features"][()]).float()
            user_summary = torch.from_numpy(group["user_summary"][()]).float()

            if "gtscore" in group:
                target_scores = torch.from_numpy(group["gtscore"][()]).float()
            else:
                target_scores = self._aggregate_user_summary(user_summary)

            change_points = (
                torch.from_numpy(group["change_points"][()]).long()
                if "change_points" in group
                else None
            )
            picks = (
                torch.from_numpy(group["picks"][()]).long()
                if "picks" in group
                else None
            )
            n_frames = (
                int(group["n_frames"][()]) if "n_frames" in group else features.shape[0]
            )

        features, target_scores, user_summary, frame_mask = self._maybe_pad(
            features, target_scores, user_summary
        )

        return VideoSummaryItem(
            video_id=video_id,
            features=features,
            frame_mask=frame_mask,
            target_scores=target_scores,
            user_summary=user_summary,
            change_points=change_points,
            picks=picks,
            n_frames=n_frames,
        )

    def _aggregate_user_summary(self, user_summary: torch.Tensor) -> torch.Tensor:
        if user_summary.dim() != 2:
            raise ValueError("user_summary must have shape (num_users, seq_len)")

        if self.aggregate == "mean":
            return user_summary.mean(dim=0)
        if self.aggregate == "median":
            return user_summary.median(dim=0).values
        if self.aggregate == "max":
            return user_summary.max(dim=0).values

        raise ValueError(f"Unknown aggregate strategy: {self.aggregate}")

    def _maybe_pad(
        self,
        features: torch.Tensor,
        target_scores: torch.Tensor,
        user_summary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        seq_len = features.shape[0]
        frame_mask = torch.ones(seq_len, dtype=torch.bool)

        if self.max_seq_len is None:
            return features, target_scores, user_summary, frame_mask

        if seq_len > self.max_seq_len:
            features = features[: self.max_seq_len]
            target_scores = target_scores[: self.max_seq_len]
            user_summary = user_summary[:, : self.max_seq_len]
            frame_mask = frame_mask[: self.max_seq_len]
        else:
            pad = self.max_seq_len - seq_len
            features = torch.cat(
                [features, torch.full((pad, features.shape[1]), self.pad_value)], dim=0
            )
            target_scores = torch.cat([target_scores, torch.zeros(pad)], dim=0)
            user_summary = torch.cat(
                [user_summary, torch.zeros(user_summary.shape[0], pad)], dim=1
            )
            frame_mask = torch.cat(
                [frame_mask, torch.zeros(pad, dtype=torch.bool)], dim=0
            )

        return features, target_scores, user_summary, frame_mask

    def _infer_feature_dim(self) -> int:
        """
        Reads the first video's feature tensor to determine the per-frame dimension.
        Raises descriptive errors when the data does not match expectations.
        """

        first_video = self.video_ids[0]
        with h5py.File(self.h5_path, "r") as h5_file:
            if first_video not in h5_file:
                raise ValueError(
                    f"Video id '{first_video}' missing from dataset file {self.h5_path}"
                )
            features = torch.from_numpy(h5_file[first_video]["features"][()])

        if features.dim() != 2:
            raise ValueError(
                "Expected precomputed features to have shape (seq_len, feature_dim) "
                f"but received tensor with shape {tuple(features.shape)} for video "
                f"'{first_video}'. Ensure you are using the ECCV16 HDF5 release."
            )

        return features.shape[1]


def video_summary_collate(batch: Sequence[VideoSummaryItem]) -> Dict[str, torch.Tensor]:
    """
    Pads a batch so that it can be stacked by DataLoader without a custom sampler.
    Returned dictionary keys align with what the training loop and summarizer need.
    """

    batch_size = len(batch)
    max_len = max(item.features.shape[0] for item in batch)
    feature_dim = batch[0].features.shape[1]
    # Different videos (especially SumMe) may have varying numbers of annotators.
    # We pad along the user dimension so that the collated tensor is rectangular.
    max_users = max(item.user_summary.shape[0] for item in batch)

    features = torch.zeros(batch_size, max_len, feature_dim)
    target_scores = torch.zeros(batch_size, max_len)
    frame_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    user_summary = torch.zeros(batch_size, max_users, max_len)

    # We keep change_points, picks and n_frames as lists (no padding) because
    # they are used only at evaluation time on a per-video basis.
    change_points_list: List[Optional[torch.Tensor]] = []
    picks_list: List[Optional[torch.Tensor]] = []
    n_frames_list: List[int] = []

    for i, item in enumerate(batch):
        seq_len = item.frame_mask.sum().item()
        features[i, :seq_len] = item.features[:seq_len]
        target_scores[i, :seq_len] = item.target_scores[:seq_len]
        frame_mask[i, :seq_len] = item.frame_mask[:seq_len]
        user_count = item.user_summary.shape[0]
        user_summary[i, :user_count, :seq_len] = item.user_summary[:, :seq_len]
        change_points_list.append(item.change_points)
        picks_list.append(item.picks)
        n_frames_list.append(item.n_frames)

    return {
        "features": features,
        "target_scores": target_scores,
        "frame_mask": frame_mask,
        "user_summary": user_summary,
        "change_points": change_points_list,
        "picks": picks_list,
        "n_frames": torch.tensor(n_frames_list, dtype=torch.long),
        "video_ids": [item.video_id for item in batch],
    }
