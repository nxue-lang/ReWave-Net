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

from mri_recon.data.dataset import FastMRISingleCoilDataset
from mri_recon.evaluation.metrics import compute_reconstruction_metrics
from mri_recon.models.unet import UNet
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net MRI baseline.")

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_psnr",
        "val_ssim",
        "val_mae",
    ]

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
    plt.ylabel("L1 Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    losses = []
    psnr_values = []
    ssim_values = []
    mae_values = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            losses.append(float(loss.item()))

            for prediction, target in zip(predictions.cpu(), targets.cpu()):
                prediction_np = prediction.squeeze(0).numpy()
                target_np = target.squeeze(0).numpy()

                metrics = compute_reconstruction_metrics(
                    prediction=prediction_np,
                    target=target_np,
                )

                psnr_values.append(metrics["psnr"])
                ssim_values.append(metrics["ssim"])
                mae_values.append(metrics["mae"])

    return {
        "val_loss": float(np.mean(losses)),
        "val_psnr": float(np.mean(psnr_values)),
        "val_ssim": float(np.mean(ssim_values)),
        "val_mae": float(np.mean(mae_values)),
    }


def save_example_reconstruction(
    model: nn.Module,
    dataset: FastMRISingleCoilDataset,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()

    sample = dataset[0]
    input_image = sample["input"].unsqueeze(0).to(device)
    target_image = sample["target"]

    with torch.no_grad():
        prediction = model(input_image).cpu().squeeze(0)

    input_np = sample["input"].squeeze(0).numpy()
    prediction_np = prediction.squeeze(0).numpy()
    target_np = target_image.squeeze(0).numpy()
    error_np = np.abs(prediction_np - target_np)

    save_image_grid(
        images=[target_np, input_np, prediction_np, error_np],
        titles=["Ground Truth", "Zero-filled", "U-Net", "Absolute Error"],
        output_path=output_path,
    )


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

    total_slices = 35
    train_indices = list(range(0, 28))
    val_indices = list(range(28, total_slices))

    train_dataset = FastMRISingleCoilDataset(
        h5_path=args.h5_path,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        slice_indices=train_indices,
    )

    val_dataset = FastMRISingleCoilDataset(
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

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    history = []
    best_val_psnr = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['val_loss']:.6f} | "
            f"val_psnr={val_metrics['val_psnr']:.4f} | "
            f"val_ssim={val_metrics['val_ssim']:.4f} | "
            f"val_mae={val_metrics['val_mae']:.6f}"
        )

        if val_metrics["val_psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["val_psnr"]
            checkpoint_path = checkpoints_dir / "unet_baseline_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_psnr": best_val_psnr,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    metrics_path = metrics_dir / "unet_training_metrics.csv"
    save_training_metrics(history, metrics_path)

    loss_curve_path = figures_dir / "unet_loss_curve.png"
    plot_loss_curve(history, loss_curve_path)

    example_path = figures_dir / "unet_reconstruction_example.png"
    save_example_reconstruction(
        model=model,
        dataset=val_dataset,
        device=device,
        output_path=example_path,
    )

    print("U-Net baseline training completed.")
    print(f"Best validation PSNR: {best_val_psnr:.4f}")
    print(f"Saved checkpoint to: {checkpoints_dir / 'unet_baseline_best.pt'}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved loss curve to: {loss_curve_path}")
    print(f"Saved example reconstruction to: {example_path}")


if __name__ == "__main__":
    main()
