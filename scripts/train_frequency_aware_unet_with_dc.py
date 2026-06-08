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
from mri_recon.models.frequency_aware_unet import FrequencyAwareComplexUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train FA-ComplexUNet with train-time hard k-space data consistency."
        )
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dc-loss-weight", type=float, default=1.0)
    parser.add_argument("--raw-loss-weight", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def channels_to_complex_torch(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, 2, H, W] real tensor to [B, H, W] complex tensor."""
    return torch.complex(x[:, 0], x[:, 1])


def complex_to_channels_torch(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, H, W] complex tensor to [B, 2, H, W] real tensor."""
    return torch.stack([x.real, x.imag], dim=1)


def fft2c_torch(image: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT for complex torch tensor with shape [B, H, W]."""
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(image, norm="ortho")
    kspace = torch.fft.fftshift(kspace, dim=(-2, -1))
    return kspace


def ifft2c_torch(kspace: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse FFT for complex torch tensor with shape [B, H, W]."""
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, norm="ortho")
    image = torch.fft.fftshift(image, dim=(-2, -1))
    return image


def apply_hard_dc_torch(
    predicted_channels: torch.Tensor,
    measured_kspace_channels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply hard k-space DC using torch operations.

    Args:
        predicted_channels: Predicted complex image as [B, 2, H, W].
        measured_kspace_channels: Measured undersampled k-space as [B, 2, H, W].
        mask: Boolean sampling mask as [B, W].

    Returns:
        Data-consistent complex image as [B, 2, H, W].
    """
    predicted_complex = channels_to_complex_torch(predicted_channels)
    measured_kspace = channels_to_complex_torch(measured_kspace_channels)

    predicted_kspace = fft2c_torch(predicted_complex)

    mask = mask.bool()
    mask = mask[:, None, :]  # [B, 1, W], broadcast over height

    corrected_kspace = torch.where(
        mask,
        measured_kspace,
        predicted_kspace,
    )

    corrected_image = ifft2c_torch(corrected_kspace)

    return complex_to_channels_torch(corrected_image)


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
    plt.ylabel("DC-aware Complex L1 Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_dc_training_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    measured_kspaces: torch.Tensor,
    masks: torch.Tensor,
    loss_fn: nn.Module,
    dc_loss_weight: float,
    raw_loss_weight: float,
) -> torch.Tensor:
    predictions = model(inputs)

    predictions_dc = apply_hard_dc_torch(
        predicted_channels=predictions,
        measured_kspace_channels=measured_kspaces,
        mask=masks,
    )

    dc_loss = loss_fn(predictions_dc, targets)
    raw_loss = loss_fn(predictions, targets)

    return dc_loss_weight * dc_loss + raw_loss_weight * raw_loss


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    dc_loss_weight: float,
    raw_loss_weight: float,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target_complex"].to(device)
            measured_kspaces = batch["measured_kspace"].to(device)
            masks = batch["mask"].to(device)

            loss = compute_dc_training_loss(
                model=model,
                inputs=inputs,
                targets=targets,
                measured_kspaces=measured_kspaces,
                masks=masks,
                loss_fn=loss_fn,
                dc_loss_weight=dc_loss_weight,
                raw_loss_weight=raw_loss_weight,
            )

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

    model = FrequencyAwareComplexUNet(
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target_complex"].to(device)
            measured_kspaces = batch["measured_kspace"].to(device)
            masks = batch["mask"].to(device)

            loss = compute_dc_training_loss(
                model=model,
                inputs=inputs,
                targets=targets,
                measured_kspaces=measured_kspaces,
                masks=masks,
                loss_fn=loss_fn,
                dc_loss_weight=args.dc_loss_weight,
                raw_loss_weight=args.raw_loss_weight,
            )

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
            dc_loss_weight=args.dc_loss_weight,
            raw_loss_weight=args.raw_loss_weight,
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

            checkpoint_path = checkpoints_dir / "frequency_aware_unet_dc_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    metrics_path = metrics_dir / "frequency_aware_unet_dc_training_metrics.csv"
    save_training_metrics(history, metrics_path)

    loss_curve_path = figures_dir / "frequency_aware_unet_dc_loss_curve.png"
    plot_loss_curve(history, loss_curve_path)

    print("FA-ComplexUNet train-time DC training completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved checkpoint to: {checkpoints_dir / 'frequency_aware_unet_dc_best.pt'}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved loss curve to: {loss_curve_path}")


if __name__ == "__main__":
    main()
