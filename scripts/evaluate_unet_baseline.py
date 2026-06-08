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
from mri_recon.models.unet import UNet
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net baseline against zero-filled reconstruction."
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def average_metric_rows(
    rows: list[dict[str, float]], method: str
) -> dict[str, float | str]:
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

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_slices = 35
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

    example_saved = False

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            predictions = model(inputs)

            for input_tensor, prediction_tensor, target_tensor in zip(
                inputs.cpu(), predictions.cpu(), targets.cpu()
            ):
                zero_filled_np = input_tensor.squeeze(0).numpy()
                prediction_np = prediction_tensor.squeeze(0).numpy()
                target_np = target_tensor.squeeze(0).numpy()

                zero_filled_metrics = compute_reconstruction_metrics(
                    prediction=zero_filled_np,
                    target=target_np,
                )

                unet_metrics = compute_reconstruction_metrics(
                    prediction=prediction_np,
                    target=target_np,
                )

                zero_filled_rows.append(zero_filled_metrics)
                unet_rows.append(unet_metrics)

                if not example_saved:
                    error_zero_filled = np.abs(zero_filled_np - target_np)
                    error_unet = np.abs(prediction_np - target_np)

                    save_image_grid(
                        images=[
                            target_np,
                            zero_filled_np,
                            prediction_np,
                            error_zero_filled,
                            error_unet,
                        ],
                        titles=[
                            "Ground Truth",
                            "Zero-filled",
                            "U-Net",
                            "ZF Error",
                            "U-Net Error",
                        ],
                        output_path=figures_dir / "unet_vs_zero_filled_example.png",
                    )

                    example_saved = True

    zero_filled_average = average_metric_rows(zero_filled_rows, "zero_filled")
    unet_average = average_metric_rows(unet_rows, "unet")

    metrics_path = metrics_dir / "unet_vs_zero_filled_metrics.csv"
    save_metrics_csv([zero_filled_average, unet_average], metrics_path)

    print("U-Net evaluation completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation slices: {val_indices}")
    print()
    print(
        f"Zero-filled | "
        f"PSNR={zero_filled_average['psnr']:.4f} | "
        f"SSIM={zero_filled_average['ssim']:.4f} | "
        f"MAE={zero_filled_average['mae']:.6f}"
    )
    print(
        f"U-Net       | "
        f"PSNR={unet_average['psnr']:.4f} | "
        f"SSIM={unet_average['ssim']:.4f} | "
        f"MAE={unet_average['mae']:.6f}"
    )
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved example to: {figures_dir / 'unet_vs_zero_filled_example.png'}")


if __name__ == "__main__":
    main()
