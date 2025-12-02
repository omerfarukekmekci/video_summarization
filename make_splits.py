import argparse
import json
from pathlib import Path
from typing import List

import h5py
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a simple K-fold splits.json file for an ECCV16-style HDF5 "
            "video summarization dataset."
        )
    )
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to eccv16_dataset_*.h5 (e.g. eccv16_dataset_summe_google_pool5.h5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("splits.json"),
        help="Output JSON file to write splits mapping into.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of splits/folds to create (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling video ids.",
    )
    return parser.parse_args()


def load_video_ids(h5_path: Path) -> List[str]:
    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found at {h5_path}. "
            "Make sure the path is correct (e.g. datasets/eccv16_dataset_summe_google_pool5.h5)."
        )

    with h5py.File(h5_path, "r") as f:
        ids = sorted(list(f.keys()))

    if not ids:
        raise ValueError(
            f"No video groups found inside {h5_path}. "
            "This file does not look like a valid ECCV16 HDF5 dataset."
        )

    return ids


def build_kfold_splits(ids: List[str], k: int, seed: int) -> dict:
    if k <= 0:
        raise ValueError(f"folds must be > 0, got {k}")

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    splits = {str(i): [] for i in range(k)}
    for idx, vid in enumerate(shuffled):
        split_idx = idx % k
        splits[str(split_idx)].append(vid)

    return splits


def main() -> None:
    args = parse_args()
    ids = load_video_ids(args.h5)
    splits = build_kfold_splits(ids, args.folds, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2))

    print(f"Found {len(ids)} videos in {args.h5}")
    print(f"Created {args.folds} splits and saved to {args.out}")
    print("Example JSON structure:")
    print(json.dumps({k: v[:3] for k, v in splits.items()}, indent=2))


if __name__ == "__main__":
    main()

