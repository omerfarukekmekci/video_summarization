import os
import subprocess
import json
from datetime import datetime

DATASETS = {
    "summe": "datasets\\eccv16_dataset_summe_google_pool5.h5",
    "tvsum": "datasets\\eccv16_dataset_tvsum_google_pool5.h5",
}

SPLIT_FILE_SUMME = "splits_summe.json"
SPLIT_FILE_TVSUM = "splits_tvsum.json"
DEVICE = "cuda"

RESULTS = {"summe": [], "tvsum": []}


def run(cmd):
    print("\n=== Running ===")
    print(" ".join(cmd))
    print("================\n")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("✗ Command failed:")
        print(result.stderr)
        return None

    print(result.stdout)
    return result.stdout


def train_and_eval(dataset_name):
    dataset_path = DATASETS[dataset_name]
    split_file = SPLIT_FILE_SUMME if dataset_name == "summe" else SPLIT_FILE_TVSUM

    print(f"\n\n==============================")
    print(f"     DATASET: {dataset_name.upper()}")
    print(f"==============================\n")

    for split in range(5):
        ckpt_name = f"checkpoints/{dataset_name}_split{split}.pth"

        # --- TRAIN ---
        train_cmd = [
            "python",
            "train.py",
            "--dataset",
            dataset_path,
            "--split-file",
            split_file,
            "--split",
            str(split),
            "--device",
            DEVICE,
            "--use-precomputed",
            "--epochs",
            "100",
            "--checkpoint",
            ckpt_name,
        ]

        print(f"\n🔵 Training {dataset_name} split {split}...")
        train_out = run(train_cmd)
        if train_out is None:
            RESULTS[dataset_name].append(("train_error", None))
            continue

        # --- EVALUATE ---
        eval_cmd = [
            "python",
            "evaluate.py",
            "--dataset",
            dataset_path,
            "--split-file",
            split_file,
            "--split",
            str(split),
            "--checkpoint",
            ckpt_name,
            "--device",
            DEVICE,
            "--batch-size",
            "1",
        ]

        print(f"\n🟢 Evaluating {dataset_name} split {split}...")
        eval_out = run(eval_cmd)
        if eval_out is None:
            RESULTS[dataset_name].append(("eval_error", None))
            continue

        # Parse a line like: "..., F-score=0.5243"
        score = None
        for line in eval_out.splitlines():
            if "F-score" in line:
                try:
                    # take the last 'F-score=...' segment if multiple appear
                    part = line.split("F-score")[-1]
                    # handles formats like "F-score=0.5243" or "F-score = 0.5243"
                    value_str = part.split("=")[-1].strip()
                    score = float(value_str)
                except (IndexError, ValueError):
                    continue
        RESULTS[dataset_name].append(score)

    print("\nDone with dataset:", dataset_name)


def print_summary():
    print("\n\n=====================================")
    print("         FINAL PERFORMANCE")
    print("=====================================\n")

    for name, scores in RESULTS.items():
        print(f"\n{name.upper()} RESULTS")
        print("-" * 40)

        valid = [s for s in scores if isinstance(s, float)]

        for i, s in enumerate(scores):
            if isinstance(s, float):
                print(f"Split {i}: {s:.4f}")
            else:
                print(f"Split {i}: ERROR")

        if len(valid) > 0:
            print(f"\nAverage: {sum(valid)/len(valid):.4f}")
        else:
            print("\nAverage: N/A (all errors)")

    print("\n=====================================\n")


def main():
    os.makedirs("checkpoints", exist_ok=True)

    train_and_eval("summe")
    train_and_eval("tvsum")

    print_summary()


if __name__ == "__main__":
    main()
