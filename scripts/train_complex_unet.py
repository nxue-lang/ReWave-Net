from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mri_recon.data.complex_dataset import FastMRIComplexSingleCoilDataset
from mri_recon.models.complex_unet import ComplexUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a complex-valued U-Net baseline."
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_training_metrics(
    rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["epoch", "train_loss", "val_loss"]
        )
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

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_indices = list(range(0, 28))
    val_indices = list(range(28, 35))

    train_dataset = FastMRIComplexSingleCoilDataset(
        h5_path=args.h5_path,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        slice_indices=train_indices,
    )

    val_dataset = FastMRIComplexSingleCoilDataset(
        h5_path=args.h5_path,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        slice_indices=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ComplexUNet(base_channels=16).to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    history = []
    best_val_loss = float("inf")

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
            dataloader=val_loader,
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
            checkpoint_path = checkpoints_dir / "complex_unet_best.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    metrics_path = metrics_dir / "complex_unet_training_metrics.csv"
    save_training_metrics(history, metrics_path)

    loss_curve_path = figures_dir / "complex_unet_loss_curve.png"
    plot_loss_curve(history, loss_curve_path)

    print("Complex U-Net training completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved checkpoint to: {checkpoints_dir / 'complex_unet_best.pt'}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved loss curve to: {loss_curve_path}")


if __name__ == "__main__":
    main()
