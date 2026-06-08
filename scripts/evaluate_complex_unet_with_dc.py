from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_recon.data.complex_dataset import (
    FastMRIComplexSingleCoilDataset,
    channels_to_complex,
)
from mri_recon.evaluation.complex_metrics import (
    complex_image_target_scaled_magnitude_pair,
    compute_complex_channel_metrics,
    compute_complex_image_metrics,
    target_scaled_magnitude_pair,
)
from mri_recon.models.complex_unet import ComplexUNet
from mri_recon.reconstruction.data_consistency import apply_hard_data_consistency
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Complex U-Net with hard k-space data consistency."
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/checkpoints/complex_unet_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def average_rows(rows: list[dict[str, float]], method: str) -> dict[str, float | str]:
    return {
        "method": method,
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
    }


def save_metrics_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "psnr", "ssim", "mae"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ComplexUNet(base_channels=16).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_slices = 35
    val_indices = list(range(28, total_slices))

    val_dataset = FastMRIComplexSingleCoilDataset(
        h5_path=args.h5_path,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        slice_indices=val_indices,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    zero_filled_rows = []
    complex_unet_rows = []
    complex_unet_dc_rows = []

    example_saved = False

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            targets_complex = batch["target_complex"]
            measured_kspaces = batch["measured_kspace"]
            masks = batch["mask"]

            predictions = model(inputs)

            for (
                input_tensor,
                prediction_tensor,
                target_complex_tensor,
                measured_kspace_tensor,
                mask_tensor,
            ) in zip(
                inputs.cpu(),
                predictions.cpu(),
                targets_complex,
                measured_kspaces,
                masks,
            ):
                input_np = input_tensor.numpy()
                prediction_np = prediction_tensor.numpy()
                target_complex_np = target_complex_tensor.numpy()

                prediction_complex = channels_to_complex(prediction_np)
                target_complex = channels_to_complex(target_complex_np)
                measured_kspace = channels_to_complex(measured_kspace_tensor.numpy())
                mask = mask_tensor.numpy().astype(bool)

                zero_filled_mag, target_mag = target_scaled_magnitude_pair(
                    prediction_channels=input_np,
                    target_channels=target_complex_np,
                )

                complex_unet_mag, _ = target_scaled_magnitude_pair(
                    prediction_channels=prediction_np,
                    target_channels=target_complex_np,
                )

                dc_complex = apply_hard_data_consistency(
                    predicted_image=prediction_complex,
                    measured_kspace=measured_kspace,
                    mask=mask,
                )

                complex_unet_dc_mag, _ = complex_image_target_scaled_magnitude_pair(
                    prediction_complex=dc_complex,
                    target_complex=target_complex,
                )

                zero_filled_metrics = compute_complex_channel_metrics(
                    prediction_channels=input_np,
                    target_channels=target_complex_np,
                )
                complex_unet_metrics = compute_complex_channel_metrics(
                    prediction_channels=prediction_np,
                    target_channels=target_complex_np,
                )
                complex_unet_dc_metrics = compute_complex_image_metrics(
                    prediction_complex=dc_complex,
                    target_complex=target_complex,
                )

                zero_filled_rows.append(zero_filled_metrics)
                complex_unet_rows.append(complex_unet_metrics)
                complex_unet_dc_rows.append(complex_unet_dc_metrics)

                if not example_saved:
                    save_image_grid(
                        images=[
                            target_mag,
                            zero_filled_mag,
                            complex_unet_mag,
                            complex_unet_dc_mag,
                            np.abs(complex_unet_dc_mag - target_mag),
                        ],
                        titles=[
                            "Ground Truth",
                            "Zero-filled",
                            "Complex U-Net",
                            "Complex U-Net + DC",
                            "DC Error",
                        ],
                        output_path=figures_dir / "complex_unet_with_dc_example.png",
                    )
                    example_saved = True

    rows = [
        average_rows(zero_filled_rows, "zero_filled"),
        average_rows(complex_unet_rows, "complex_unet"),
        average_rows(complex_unet_dc_rows, "complex_unet_dc"),
    ]

    metrics_path = metrics_dir / "complex_unet_with_dc_metrics.csv"
    save_metrics_csv(rows, metrics_path)

    print("Complex U-Net + Data Consistency evaluation completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation slices: {val_indices}")
    print()

    for row in rows:
        print(
            f"{row['method']:18s} | "
            f"PSNR={row['psnr']:.4f} | "
            f"SSIM={row['ssim']:.4f} | "
            f"MAE={row['mae']:.6f}"
        )

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved example to: {figures_dir / 'complex_unet_with_dc_example.png'}")


if __name__ == "__main__":
    main()
