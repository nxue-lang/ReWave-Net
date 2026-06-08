from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mri_recon.data.multi_file_complex_dataset import (
    MultiFileFastMRIComplexSingleCoilDataset,
)
from mri_recon.models.kan_frequency_aware_unet import KANFrequencyAwareComplexUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KAN-gated FA-ComplexUNet on multiple fastMRI files."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/knee_singlecoil_val",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--middle-slice-margin", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs")

    args = parser.parse_args()
    if args.mask_seed is None:
        args.mask_seed = args.seed

    return args


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_files(
    h5_paths: list[Path],
    train_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    paths = list(h5_paths)
    rng.shuffle(paths)

    split_index = int(len(paths) * train_ratio)

    train_paths = paths[:split_index]
    test_paths = paths[split_index:]

    return train_paths, test_paths


def save_file_split(
    train_paths: list[Path],
    test_paths: list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("split,file_path\n")
        for path in train_paths:
            f.write(f"train,{path.as_posix()}\n")
        for path in test_paths:
            f.write(f"test,{path.as_posix()}\n")


def save_training_metrics(
    rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "train_loss", "val_loss"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curve(
    rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    val_loss = [row["val_loss"] for row in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Complex L1 Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target_complex"].to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            losses.append(float(loss.item()))

    return float(np.mean(losses))


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    splits_dir = output_dir / "splits"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    h5_paths = sorted(data_dir.glob("*.h5"))

    if len(h5_paths) == 0:
        raise FileNotFoundError(f"No h5 files found in {data_dir}")

    train_paths, test_paths = split_files(
        h5_paths=h5_paths,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    split_path = splits_dir / f"multifile_split_seed{args.seed}.csv"
    save_file_split(train_paths, test_paths, split_path)

    print(f"Total files: {len(h5_paths)}")
    print(f"Train files: {len(train_paths)}")
    print(f"Test files: {len(test_paths)}")
    print(f"Saved split to: {split_path}")

    train_dataset = MultiFileFastMRIComplexSingleCoilDataset(
        h5_paths=train_paths,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        use_middle_slices_only=True,
        middle_slice_margin=args.middle_slice_margin,
        mask_seed=args.mask_seed,
    )

    test_dataset = MultiFileFastMRIComplexSingleCoilDataset(
        h5_paths=test_paths,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        use_middle_slices_only=True,
        middle_slice_margin=args.middle_slice_margin,
        mask_seed=args.mask_seed,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = KANFrequencyAwareComplexUNet(
        base_channels=args.base_channels,
    ).to(device)

    loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    history = []
    best_val_loss = float("inf")

    checkpoint_name = (
        f"kan_frequency_aware_unet_multifile_acc{args.acceleration}_best.pt"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target_complex"].to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))

        val_loss = evaluate_loss(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = checkpoints_dir / checkpoint_name
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                    "train_files": [path.as_posix() for path in train_paths],
                    "test_files": [path.as_posix() for path in test_paths],
                },
                checkpoint_path,
            )

    metrics_filename = (
        f"kan_frequency_aware_unet_multifile_acc{args.acceleration}"
        "_training_metrics.csv"
    )
    metrics_path = metrics_dir / metrics_filename
    save_training_metrics(history, metrics_path)

    loss_curve_path = (
        figures_dir
        / f"kan_frequency_aware_unet_multifile_acc{args.acceleration}_loss_curve.png"
    )
    plot_loss_curve(history, loss_curve_path)

    print("Multi-file KAN-gated FA-ComplexUNet training completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved checkpoint to: {checkpoints_dir / checkpoint_name}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved loss curve to: {loss_curve_path}")


if __name__ == "__main__":
    main()
