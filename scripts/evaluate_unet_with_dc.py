from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_recon.data.dataset import FastMRISingleCoilDataset
from mri_recon.evaluation.metrics import compute_reconstruction_metrics
from mri_recon.io import load_kspace_from_h5
from mri_recon.models.unet import UNet
from mri_recon.reconstruction.data_consistency import apply_hard_data_consistency
from mri_recon.sampling import (
    apply_undersampling_mask,
    create_cartesian_undersampling_mask,
)
from mri_recon.transforms import center_crop, center_pad, normalize_to_unit_range
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net with hard k-space data consistency."
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/checkpoints/unet_baseline_best.pt",
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


def save_metrics_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "psnr", "ssim", "mae"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average_rows(rows: list[dict[str, float]], method: str) -> dict[str, float | str]:
    return {
        "method": method,
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    kspace = load_kspace_from_h5(args.h5_path)
    total_slices = kspace.shape[0]
    val_indices = list(range(28, total_slices))

    val_dataset = FastMRISingleCoilDataset(
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
    unet_rows = []
    unet_dc_rows = []

    example_saved = False

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            slice_indices = batch["slice_index"]

            predictions = model(inputs)

            for (
                input_tensor,
                prediction_tensor,
                target_tensor,
                slice_index_tensor,
            ) in zip(inputs.cpu(), predictions.cpu(), targets.cpu(), slice_indices):
                slice_index = int(slice_index_tensor)

                zero_filled_np = input_tensor.squeeze(0).numpy()
                unet_np = prediction_tensor.squeeze(0).numpy()
                target_np = target_tensor.squeeze(0).numpy()

                kspace_slice = kspace[slice_index]

                mask = create_cartesian_undersampling_mask(
                    width=kspace_slice.shape[-1],
                    acceleration=args.acceleration,
                    center_fraction=args.center_fraction,
                    seed=0,
                )

                undersampled_kspace = apply_undersampling_mask(
                    kspace=kspace_slice,
                    mask=mask,
                )

                # Pad U-Net output back to full size before FFT/k-space DC.
                unet_padded = center_pad(
                    unet_np,
                    target_shape=kspace_slice.shape,
                ).astype(np.float32)

                unet_complex = unet_padded.astype(np.complex64)

                dc_complex = apply_hard_data_consistency(
                    predicted_image=unet_complex,
                    measured_kspace=undersampled_kspace,
                    mask=mask,
                )

                dc_magnitude = normalize_to_unit_range(dc_complex)
                dc_cropped = center_crop(dc_magnitude, target_shape=target_np.shape)
                dc_cropped = normalize_to_unit_range(dc_cropped).astype(np.float32)

                zero_filled_metrics = compute_reconstruction_metrics(
                    prediction=zero_filled_np,
                    target=target_np,
                )
                unet_metrics = compute_reconstruction_metrics(
                    prediction=unet_np,
                    target=target_np,
                )
                unet_dc_metrics = compute_reconstruction_metrics(
                    prediction=dc_cropped,
                    target=target_np,
                )

                zero_filled_rows.append(zero_filled_metrics)
                unet_rows.append(unet_metrics)
                unet_dc_rows.append(unet_dc_metrics)

                if not example_saved:
                    save_image_grid(
                        images=[
                            target_np,
                            zero_filled_np,
                            unet_np,
                            dc_cropped,
                            np.abs(dc_cropped - target_np),
                        ],
                        titles=[
                            "Ground Truth",
                            "Zero-filled",
                            "U-Net",
                            "U-Net + DC",
                            "DC Error",
                        ],
                        output_path=figures_dir / "unet_with_dc_example.png",
                    )
                    example_saved = True

    rows = [
        average_rows(zero_filled_rows, "zero_filled"),
        average_rows(unet_rows, "unet"),
        average_rows(unet_dc_rows, "unet_dc"),
    ]

    metrics_path = metrics_dir / "unet_with_dc_metrics.csv"
    save_metrics_csv(rows, metrics_path)

    print("U-Net + Data Consistency evaluation completed.")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Validation slices: {val_indices}")

    for row in rows:
        print(
            f"{row['method']:12s} | "
            f"PSNR={row['psnr']:.4f} | "
            f"SSIM={row['ssim']:.4f} | "
            f"MAE={row['mae']:.6f}"
        )

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved example to: {figures_dir / 'unet_with_dc_example.png'}")


if __name__ == "__main__":
    main()
