from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mri_recon.data.multi_file_complex_dataset import (
    MultiFileFastMRIComplexSingleCoilDataset,
)
from mri_recon.evaluation.complex_metrics import (
    compute_complex_channel_metrics,
    target_scaled_magnitude_pair,
)
from mri_recon.models.unrolled_frequency_aware import (
    UnrolledComplexUNetRecon,
    UnrolledFrequencyAwareRecon,
    UnrolledKANFrequencyAwareRecon,
    UnrolledResidualConditionedWaveletRecon,
)
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an unrolled frequency-aware MRI reconstruction model."
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/checkpoints/unrolled_residual_wavelet_recon_c5_acc4_best.pt",
    )
    parser.add_argument("--data-dir", type=str, default="data/knee_singlecoil_val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_checkpoint_args(checkpoint_args: dict[str, object]) -> nn.Module:
    model_type = str(checkpoint_args.get("model_type", "kan"))
    num_cascades = int(checkpoint_args.get("num_cascades", 5))
    base_channels = int(checkpoint_args.get("base_channels", 8))
    shared_denoiser = not bool(checkpoint_args.get("unshared_denoiser", False))
    initial_dc_weight = float(checkpoint_args.get("initial_dc_weight", 0.1))

    model_kwargs = {
        "num_cascades": num_cascades,
        "base_channels": base_channels,
        "shared_denoiser": shared_denoiser,
        "initial_dc_weight": initial_dc_weight,
    }

    if model_type == "complex":
        return UnrolledComplexUNetRecon(**model_kwargs)
    if model_type == "fa":
        return UnrolledFrequencyAwareRecon(**model_kwargs)
    if model_type == "residual_wavelet":
        return UnrolledResidualConditionedWaveletRecon(**model_kwargs)

    return UnrolledKANFrequencyAwareRecon(**model_kwargs)


def method_name_from_checkpoint_args(checkpoint_args: dict[str, object]) -> str:
    model_type = str(checkpoint_args.get("model_type", "kan"))
    return {
        "complex": "unrolled_complex_unet",
        "fa": "unrolled_fa_complex_unet",
        "kan": "unrolled_kan_fa_complex_unet",
        "residual_wavelet": "unrolled_residual_conditioned_wavelet",
    }.get(model_type, f"unrolled_{model_type}")


def average_metric_rows(
    rows: list[dict[str, float]], method: str
) -> dict[str, float | str]:
    return {
        "method": method,
        "psnr_mean": float(np.mean([row["psnr"] for row in rows])),
        "psnr_std": float(np.std([row["psnr"] for row in rows])),
        "ssim_mean": float(np.mean([row["ssim"] for row in rows])),
        "ssim_std": float(np.std([row["ssim"] for row in rows])),
        "mae_mean": float(np.mean([row["mae"] for row in rows])),
        "mae_std": float(np.std([row["mae"] for row in rows])),
    }


def save_summary_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "mae_mean",
        "mae_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_per_slice_csv(
    rows: list[dict[str, float | int | str]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "file_path", "slice_index", "psnr", "ssim", "mae"]
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

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    print(f"Using device: {device}")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Using AMP: {use_amp}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})

    model = build_model_from_checkpoint_args(checkpoint_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    method_name = method_name_from_checkpoint_args(checkpoint_args)

    test_files = checkpoint.get("test_files")
    if not test_files:
        data_dir = Path(args.data_dir)
        test_files = [path.as_posix() for path in sorted(data_dir.glob("*.h5"))]

    test_paths = [Path(path) for path in test_files]
    print(f"Evaluation files: {len(test_paths)}")

    acceleration = int(checkpoint_args.get("acceleration", 4))
    center_fraction = float(checkpoint_args.get("center_fraction", 0.08))
    middle_slice_margin = int(checkpoint_args.get("middle_slice_margin", 5))
    checkpoint_mask_seed = checkpoint_args.get("mask_seed")
    mask_seed = None if checkpoint_mask_seed is None else int(checkpoint_mask_seed)

    dataset = MultiFileFastMRIComplexSingleCoilDataset(
        h5_paths=test_paths,
        acceleration=acceleration,
        center_fraction=center_fraction,
        use_middle_slices_only=True,
        middle_slice_margin=middle_slice_margin,
        mask_seed=mask_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    zero_filled_rows = []
    unrolled_rows = []
    per_slice_rows = []
    example_saved = False

    with torch.no_grad():
        iterator = dataloader
        if not args.disable_progress:
            iterator = tqdm(dataloader, desc="Evaluating", leave=False)

        for batch in iterator:
            inputs = batch["input"].to(device)
            targets = batch["target_complex"].to(device)
            measured_kspaces = batch["measured_kspace"].to(device)
            masks = batch["mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(
                    image=inputs,
                    measured_kspace=measured_kspaces,
                    mask=masks,
                )

            for sample_index, (
                input_tensor,
                prediction_tensor,
                target_tensor,
                file_path,
                slice_index,
            ) in enumerate(
                zip(
                    inputs.cpu(),
                    predictions.cpu(),
                    targets.cpu(),
                    batch["file_path"],
                    batch["slice_index"],
                )
            ):
                input_np = input_tensor.numpy()
                prediction_np = prediction_tensor.numpy()
                target_np = target_tensor.numpy()

                zero_filled_metrics = compute_complex_channel_metrics(
                    prediction_channels=input_np,
                    target_channels=target_np,
                )
                unrolled_metrics = compute_complex_channel_metrics(
                    prediction_channels=prediction_np,
                    target_channels=target_np,
                )

                zero_filled_rows.append(zero_filled_metrics)
                unrolled_rows.append(unrolled_metrics)

                slice_index_value = int(slice_index)
                for method, metrics in [
                    ("zero_filled", zero_filled_metrics),
                    (method_name, unrolled_metrics),
                ]:
                    per_slice_rows.append(
                        {
                            "method": method,
                            "file_path": str(file_path),
                            "slice_index": slice_index_value,
                            **metrics,
                        }
                    )

                if not example_saved and sample_index == 0:
                    zero_mag, target_mag = target_scaled_magnitude_pair(
                        prediction_channels=input_np,
                        target_channels=target_np,
                    )
                    pred_mag, _ = target_scaled_magnitude_pair(
                        prediction_channels=prediction_np,
                        target_channels=target_np,
                    )
                    error_mag = np.abs(pred_mag - target_mag)

                    save_image_grid(
                        images=[
                            target_mag,
                            zero_mag,
                            pred_mag,
                            error_mag,
                        ],
                        titles=[
                            "Target",
                            "Zero-filled",
                            method_name,
                            "Absolute Error",
                        ],
                        output_path=figures_dir
                        / "unrolled_frequency_aware_recon_example.png",
                    )
                    example_saved = True

    summary_rows = [
        average_metric_rows(zero_filled_rows, "zero_filled"),
        average_metric_rows(unrolled_rows, method_name),
    ]

    summary_path = metrics_dir / f"{method_name}_metrics.csv"
    per_slice_path = metrics_dir / f"{method_name}_per_slice_metrics.csv"
    save_summary_csv(summary_rows, summary_path)
    save_per_slice_csv(per_slice_rows, per_slice_path)

    print("Unrolled frequency-aware evaluation completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    for row in summary_rows:
        print(
            f"{row['method']:26s} | "
            f"PSNR={row['psnr_mean']:.4f} +/- {row['psnr_std']:.4f} | "
            f"SSIM={row['ssim_mean']:.4f} +/- {row['ssim_std']:.4f} | "
            f"MAE={row['mae_mean']:.6f} +/- {row['mae_std']:.6f}"
        )
    print(f"Saved summary metrics to: {summary_path}")
    print(f"Saved per-slice metrics to: {per_slice_path}")


if __name__ == "__main__":
    main()
