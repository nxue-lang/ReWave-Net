from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import torch

METRICS_DIR = Path("outputs/metrics")
CHECKPOINTS_DIR = Path("outputs/checkpoints")


METHOD_FILES = {
    "zero_filled": [
        "unrolled_kan_fa_complex_unet_metrics.csv",
        "unrolled_frequency_aware_recon_metrics.csv",
        "unrolled_fa_complex_unet_metrics.csv",
        "unrolled_complex_unet_metrics.csv",
    ],
    "unrolled_complex_unet": ["unrolled_complex_unet_metrics.csv"],
    "unrolled_fa_complex_unet": ["unrolled_fa_complex_unet_metrics.csv"],
    "unrolled_kan_fa_complex_unet": [
        "unrolled_kan_fa_complex_unet_metrics.csv",
        "unrolled_frequency_aware_recon_metrics.csv",
    ],
}


CHECKPOINT_FILES = {
    "unrolled_complex_unet": [
        "unrolled_complex_recon_c5_acc4_best.pt",
        "unrolled_complex_recon_c3_acc4_best.pt",
        "unrolled_complex_unet_recon_c5_acc4_best.pt",
        "unrolled_complex_unet_recon_c3_acc4_best.pt",
    ],
    "unrolled_fa_complex_unet": [
        "unrolled_fa_recon_c5_acc4_best.pt",
        "unrolled_fa_recon_c3_acc4_best.pt",
        "unrolled_fa_complex_unet_recon_c5_acc4_best.pt",
        "unrolled_fa_complex_unet_recon_c3_acc4_best.pt",
    ],
    "unrolled_kan_fa_complex_unet": [
        "unrolled_kan_recon_c5_acc4_best.pt",
        "unrolled_kan_recon_c3_acc4_best.pt",
        "unrolled_kan_frequency_aware_recon_c5_acc4_best.pt",
        "unrolled_kan_frequency_aware_recon_c3_acc4_best.pt",
    ],
}


CONFIG_FIELDS = [
    "model_type",
    "num_cascades",
    "base_channels",
    "epochs",
    "acceleration",
    "center_fraction",
    "seed",
    "mask_seed",
    "middle_slice_margin",
    "train_ratio",
]


FAIRNESS_FIELDS = [
    "num_cascades",
    "base_channels",
    "epochs",
    "acceleration",
    "center_fraction",
    "seed",
    "mask_seed",
    "middle_slice_margin",
    "train_ratio",
    "test_files_digest",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def find_method_row(method: str) -> dict[str, str] | None:
    for filename in METHOD_FILES[method]:
        path = METRICS_DIR / filename
        if not path.exists():
            continue

        rows = read_csv_rows(path)
        for row in rows:
            if row["method"] == method:
                return row

        # Compatibility for the earlier generic unrolled output name.
        if method == "unrolled_kan_fa_complex_unet":
            for row in rows:
                if row["method"] == "unrolled_frequency_aware":
                    row = dict(row)
                    row["method"] = method
                    return row

        if method == "zero_filled":
            for row in rows:
                if row["method"] == "zero_filled":
                    return row

    return None


def list_digest(values: list[str]) -> str:
    if not values:
        return ""

    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=6).hexdigest()


def find_checkpoint_metadata(method: str) -> dict[str, Any]:
    for filename in CHECKPOINT_FILES.get(method, []):
        checkpoint_path = CHECKPOINTS_DIR / filename
        if not checkpoint_path.exists():
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = checkpoint.get("args", {})
        train_files = checkpoint.get("train_files", [])
        test_files = checkpoint.get("test_files", [])

        metadata: dict[str, Any] = {
            "checkpoint": checkpoint_path.as_posix(),
            "train_file_count": len(train_files),
            "test_file_count": len(test_files),
            "train_files_digest": list_digest(train_files),
            "test_files_digest": list_digest(test_files),
        }

        for field in CONFIG_FIELDS:
            value = args.get(field, "")
            if field == "mask_seed" and value == "" and args.get("seed") is not None:
                value = args["seed"]
            metadata[field] = value

        return metadata

    return {
        "checkpoint": "",
        "train_file_count": "",
        "test_file_count": "",
        "train_files_digest": "",
        "test_files_digest": "",
        **{field: "" for field in CONFIG_FIELDS},
    }


def format_float(value: str | float) -> float:
    return float(value)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def config_status(
    method: str,
    metadata: dict[str, Any],
    reference_metadata: dict[str, Any] | None,
) -> str:
    if method == "zero_filled":
        return "baseline_from_metrics"

    if not metadata.get("checkpoint"):
        return "missing_checkpoint"

    if reference_metadata is None:
        return "reference"

    mismatches = []
    for field in FAIRNESS_FIELDS:
        left_value = format_value(metadata.get(field))
        right_value = format_value(reference_metadata.get(field))
        if left_value != right_value:
            mismatches.append(field)

    if mismatches:
        return "mismatch:" + ";".join(mismatches)

    return "matched"


def build_summary_rows() -> list[dict[str, str | float]]:
    zero_row = find_method_row("zero_filled")
    if zero_row is None:
        raise FileNotFoundError("No zero_filled row found in unrolled metrics files.")

    zero_psnr = format_float(zero_row["psnr_mean"])
    zero_ssim = format_float(zero_row["ssim_mean"])
    zero_mae = format_float(zero_row["mae_mean"])

    metadata_by_method = {
        method: find_checkpoint_metadata(method)
        for method in METHOD_FILES
        if method != "zero_filled"
    }
    reference_metadata = next(
        (
            metadata
            for metadata in metadata_by_method.values()
            if metadata.get("checkpoint")
        ),
        None,
    )

    output_rows = []
    for method in METHOD_FILES:
        row = find_method_row(method)
        metadata = metadata_by_method.get(method, {})
        status = config_status(method, metadata, reference_metadata)

        if row is None:
            output_rows.append(
                {
                    "method": method,
                    "result_status": "missing",
                    "config_status": status,
                    "psnr_mean": "",
                    "psnr_std": "",
                    "ssim_mean": "",
                    "ssim_std": "",
                    "mae_mean": "",
                    "mae_std": "",
                    "delta_psnr": "",
                    "delta_ssim": "",
                    "delta_mae": "",
                    **metadata,
                }
            )
            continue

        psnr = format_float(row["psnr_mean"])
        ssim = format_float(row["ssim_mean"])
        mae = format_float(row["mae_mean"])

        output_rows.append(
            {
                "method": method,
                "result_status": "available",
                "config_status": status,
                "psnr_mean": psnr,
                "psnr_std": format_float(row["psnr_std"]),
                "ssim_mean": ssim,
                "ssim_std": format_float(row["ssim_std"]),
                "mae_mean": mae,
                "mae_std": format_float(row["mae_std"]),
                "delta_psnr": psnr - zero_psnr,
                "delta_ssim": ssim - zero_ssim,
                "delta_mae": mae - zero_mae,
                **metadata,
            }
        )

    return output_rows


def save_summary(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "result_status",
        "config_status",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "mae_mean",
        "mae_std",
        "delta_psnr",
        "delta_ssim",
        "delta_mae",
        "checkpoint",
        *CONFIG_FIELDS,
        "train_file_count",
        "test_file_count",
        "train_files_digest",
        "test_files_digest",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | float]]) -> None:
    print("Unrolled reconstruction ablation summary")
    print()
    for row in rows:
        if row["result_status"] == "missing":
            print(f"{row['method']:30s} | missing | {row['config_status']}")
            continue

        print(
            f"{row['method']:30s} | "
            f"PSNR={float(row['psnr_mean']):.4f} "
            f"({float(row['delta_psnr']):+.4f}) | "
            f"SSIM={float(row['ssim_mean']):.4f} "
            f"({float(row['delta_ssim']):+.4f}) | "
            f"MAE={float(row['mae_mean']):.6f} "
            f"({float(row['delta_mae']):+.6f}) | "
            f"{row['config_status']}"
        )

    mismatched = [
        row for row in rows if str(row.get("config_status", "")).startswith("mismatch")
    ]
    if mismatched:
        print()
        print("Config warning:")
        print(
            "Some rows are useful trend evidence but are not strict fair comparisons. "
            "Use matched settings before making final paper claims."
        )


def main() -> None:
    rows = build_summary_rows()
    output_path = METRICS_DIR / "unrolled_ablation_summary.csv"
    save_summary(rows, output_path)
    print_summary(rows)
    print()
    print(f"Saved ablation summary to: {output_path}")


if __name__ == "__main__":
    main()
