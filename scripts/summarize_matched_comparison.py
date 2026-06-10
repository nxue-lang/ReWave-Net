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
        "rewave_net_metrics.csv",
        "unrolled_residual_conditioned_wavelet_metrics.csv",
        "unrolled_complex_unet_metrics.csv",
    ],
    "unrolled_complex_unet": ["unrolled_complex_unet_metrics.csv"],
    "rewave_net": [
        "rewave_net_metrics.csv",
        "unrolled_residual_conditioned_wavelet_metrics.csv",
    ],
}

CHECKPOINT_FILES = {
    "unrolled_complex_unet": [
        "complex_c5_acc4_best.pt",
        "unrolled_complex_recon_c5_acc4_best.pt",
    ],
    "rewave_net": [
        "rewave_c5_acc4_best.pt",
        "unrolled_residual_wavelet_recon_c5_acc4_best.pt",
    ],
}

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


def read_method_row(method: str) -> dict[str, str] | None:
    aliases = {
        "rewave_net": {"rewave_net", "unrolled_residual_conditioned_wavelet"},
        "unrolled_complex_unet": {"unrolled_complex_unet"},
        "zero_filled": {"zero_filled"},
    }
    for filename in METHOD_FILES[method]:
        path = METRICS_DIR / filename
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                if row["method"] in aliases[method]:
                    return row
    return None


def list_digest(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=6).hexdigest() if values else ""


def checkpoint_metadata(method: str) -> dict[str, Any]:
    for filename in CHECKPOINT_FILES.get(method, []):
        path = CHECKPOINTS_DIR / filename
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu")
        args = checkpoint.get("args", {})
        return {
            "checkpoint": path.as_posix(),
            **{field: args.get(field, "") for field in FAIRNESS_FIELDS[:-1]},
            "test_files_digest": list_digest(checkpoint.get("test_files", [])),
        }
    return {"checkpoint": "", **{field: "" for field in FAIRNESS_FIELDS}}


def match_status(metadata: dict[str, Any], reference: dict[str, Any]) -> str:
    if not metadata["checkpoint"]:
        return "missing_checkpoint"
    mismatches = [
        field
        for field in FAIRNESS_FIELDS
        if str(metadata.get(field, "")) != str(reference.get(field, ""))
    ]
    return "matched" if not mismatches else "mismatch:" + ";".join(mismatches)


def main() -> None:
    rows = {method: read_method_row(method) for method in METHOD_FILES}
    zero = rows["zero_filled"]
    if zero is None:
        raise FileNotFoundError("No zero-filled metrics row found.")

    metadata = {
        method: checkpoint_metadata(method)
        for method in ("unrolled_complex_unet", "rewave_net")
    }
    reference = metadata["rewave_net"]

    print("Matched ReWave-Net comparison")
    print()
    for method in METHOD_FILES:
        row = rows[method]
        if row is None:
            print(f"{method:24s} | missing")
            continue
        status = (
            "baseline_from_metrics"
            if method == "zero_filled"
            else match_status(metadata[method], reference)
        )
        print(
            f"{method:24s} | "
            f"PSNR={float(row['psnr_mean']):.4f} | "
            f"SSIM={float(row['ssim_mean']):.4f} | "
            f"MAE={float(row['mae_mean']):.6f} | "
            f"{status}"
        )


if __name__ == "__main__":
    main()
