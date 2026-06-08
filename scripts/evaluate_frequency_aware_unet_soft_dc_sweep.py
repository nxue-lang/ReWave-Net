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
    compute_complex_image_metrics,
    target_scaled_magnitude_pair,
)
from mri_recon.models.frequency_aware_unet import FrequencyAwareComplexUNet
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FA-ComplexUNet with soft k-space DC sweep."
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/checkpoints/frequency_aware_unet_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fft2c_numpy(image: np.ndarray) -> np.ndarray:
    image = np.fft.ifftshift(image, axes=(-2, -1))
    kspace = np.fft.fft2(image, norm="ortho")
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))
    return kspace


def ifft2c_numpy(kspace: np.ndarray) -> np.ndarray:
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, norm="ortho")
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


def apply_soft_data_consistency(
    predicted_image: np.ndarray,
    measured_kspace: np.ndarray,
    mask: np.ndarray,
    dc_strength: float,
) -> np.ndarray:
    """Apply soft k-space data consistency.

    Formula:
        k_dc = k_pred + lambda * mask * (k_measured - k_pred)

    lambda = 0.0 means no DC.
    lambda = 1.0 equals hard DC.
    """
    predicted_kspace = fft2c_numpy(predicted_image)

    expanded_mask = mask.reshape(1, mask.shape[0])
    corrected_kspace = predicted_kspace + dc_strength * expanded_mask * (
        measured_kspace - predicted_kspace
    )

    return ifft2c_numpy(corrected_kspace)


def average_rows(rows: list[dict[str, float]], method: str) -> dict[str, float | str]:
    return {
        "method": method,
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
    }


def save_metrics_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "dc_strength", "psnr", "ssim", "mae"]

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

    model = FrequencyAwareComplexUNet(
        base_channels=args.base_channels,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dc_strengths = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    val_indices = list(range(28, 35))

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

    metrics_by_strength: dict[float, list[dict[str, float]]] = {
        strength: [] for strength in dc_strengths
    }

    example_images: dict[float, np.ndarray] = {}
    example_target = None
    example_zero_filled = None

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
                target_complex_np = target_complex_tensor.numpy()

                prediction_complex = channels_to_complex(prediction_tensor.numpy())
                target_complex = channels_to_complex(target_complex_np)
                measured_kspace = channels_to_complex(measured_kspace_tensor.numpy())
                mask = mask_tensor.numpy().astype(bool)

                zero_filled_mag, target_mag = target_scaled_magnitude_pair(
                    prediction_channels=input_np,
                    target_channels=target_complex_np,
                )

                for strength in dc_strengths:
                    soft_dc_complex = apply_soft_data_consistency(
                        predicted_image=prediction_complex,
                        measured_kspace=measured_kspace,
                        mask=mask,
                        dc_strength=strength,
                    )

                    soft_dc_mag, _ = complex_image_target_scaled_magnitude_pair(
                        prediction_complex=soft_dc_complex,
                        target_complex=target_complex,
                    )

                    metrics = compute_complex_image_metrics(
                        prediction_complex=soft_dc_complex,
                        target_complex=target_complex,
                    )
                    metrics_by_strength[strength].append(metrics)

                    if example_target is None:
                        example_target = target_mag
                        example_zero_filled = zero_filled_mag

                    if len(example_images) < len(dc_strengths):
                        example_images[strength] = soft_dc_mag

    summary_rows = []

    for strength in dc_strengths:
        avg = average_rows(
            metrics_by_strength[strength],
            method="fa_complex_unet_soft_dc",
        )
        avg["dc_strength"] = strength
        summary_rows.append(avg)

    metrics_path = metrics_dir / "frequency_aware_unet_soft_dc_sweep_metrics.csv"
    save_metrics_csv(summary_rows, metrics_path)

    print("FA-ComplexUNet soft DC sweep completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation slices: {val_indices}")
    print()

    best_by_psnr = max(summary_rows, key=lambda row: float(row["psnr"]))
    best_by_ssim = max(summary_rows, key=lambda row: float(row["ssim"]))

    for row in summary_rows:
        print(
            f"lambda={float(row['dc_strength']):.2f} | "
            f"PSNR={float(row['psnr']):.4f} | "
            f"SSIM={float(row['ssim']):.4f} | "
            f"MAE={float(row['mae']):.6f}"
        )

    print()
    print(
        f"Best PSNR: lambda={float(best_by_psnr['dc_strength']):.2f}, "
        f"PSNR={float(best_by_psnr['psnr']):.4f}"
    )
    print(
        f"Best SSIM: lambda={float(best_by_ssim['dc_strength']):.2f}, "
        f"SSIM={float(best_by_ssim['ssim']):.4f}"
    )

    if example_target is not None and example_zero_filled is not None:
        selected_strengths = [0.0, 0.05, 0.10, 0.30, 1.0]
        images = [example_target, example_zero_filled]
        titles = ["Ground Truth", "Zero-filled"]

        for strength in selected_strengths:
            closest_strength = min(
                example_images.keys(),
                key=lambda key: abs(key - strength),
            )
            images.append(example_images[closest_strength])
            titles.append(f"Soft DC lambda={closest_strength:.2f}")

        save_image_grid(
            images=images,
            titles=titles,
            output_path=figures_dir / "frequency_aware_unet_soft_dc_sweep_example.png",
        )

    print(f"Saved metrics to: {metrics_path}")
    print(
        "Saved example to: "
        f"{figures_dir / 'frequency_aware_unet_soft_dc_sweep_example.png'}"
    )


if __name__ == "__main__":
    main()
