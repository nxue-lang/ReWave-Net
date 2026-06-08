from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a report for the Unrolled Complex U-Net baseline."
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="outputs/metrics/unrolled_complex_unet_metrics.csv",
    )
    parser.add_argument(
        "--per-slice-path",
        type=str,
        default="outputs/metrics/unrolled_complex_unet_per_slice_metrics.csv",
    )
    parser.add_argument(
        "--ablation-path",
        type=str,
        default="outputs/metrics/unrolled_ablation_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="docs/unrolled_complex_unet_baseline_report.md",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def find_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("method") == method:
            return row
    raise ValueError(f"No row found for method '{method}'.")


def metric_delta(
    method_row: dict[str, str],
    baseline_row: dict[str, str],
    metric: str,
) -> float:
    return float(method_row[f"{metric}_mean"]) - float(baseline_row[f"{metric}_mean"])


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def summarize_per_slice(rows: list[dict[str, str]]) -> dict[str, int | float]:
    by_sample: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["file_path"], int(row["slice_index"]))
        by_sample.setdefault(key, {})[row["method"]] = row

    paired_samples = [
        sample
        for sample in by_sample.values()
        if "zero_filled" in sample and "unrolled_complex_unet" in sample
    ]
    if not paired_samples:
        return {
            "slices": 0,
            "psnr_improved": 0,
            "ssim_improved": 0,
            "mae_improved": 0,
            "psnr_delta_mean": 0.0,
            "ssim_delta_mean": 0.0,
            "mae_delta_mean": 0.0,
        }

    psnr_deltas = []
    ssim_deltas = []
    mae_deltas = []
    for sample in paired_samples:
        zero = sample["zero_filled"]
        method = sample["unrolled_complex_unet"]
        psnr_deltas.append(float(method["psnr"]) - float(zero["psnr"]))
        ssim_deltas.append(float(method["ssim"]) - float(zero["ssim"]))
        mae_deltas.append(float(method["mae"]) - float(zero["mae"]))

    return {
        "slices": len(paired_samples),
        "psnr_improved": sum(delta > 0 for delta in psnr_deltas),
        "ssim_improved": sum(delta > 0 for delta in ssim_deltas),
        "mae_improved": sum(delta < 0 for delta in mae_deltas),
        "psnr_delta_mean": sum(psnr_deltas) / len(psnr_deltas),
        "ssim_delta_mean": sum(ssim_deltas) / len(ssim_deltas),
        "mae_delta_mean": sum(mae_deltas) / len(mae_deltas),
    }


def format_summary_row(row: dict[str, str]) -> str:
    return (
        f"{float(row['psnr_mean']):.4f} +/- {float(row['psnr_std']):.4f} | "
        f"{float(row['ssim_mean']):.4f} +/- {float(row['ssim_std']):.4f} | "
        f"{float(row['mae_mean']):.6f} +/- {float(row['mae_std']):.6f}"
    )


def optional_ablation_metadata(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    rows = read_csv_rows(path)
    for row in rows:
        if row.get("method") == "unrolled_complex_unet":
            return row
    return {}


def build_report(
    metrics_path: Path,
    per_slice_path: Path,
    ablation_path: Path,
) -> str:
    metrics_rows = read_csv_rows(metrics_path)
    per_slice_rows = read_csv_rows(per_slice_path)
    zero = find_row(metrics_rows, "zero_filled")
    baseline = find_row(metrics_rows, "unrolled_complex_unet")
    per_slice = summarize_per_slice(per_slice_rows)
    metadata = optional_ablation_metadata(ablation_path)

    psnr_delta = metric_delta(baseline, zero, "psnr")
    ssim_delta = metric_delta(baseline, zero, "ssim")
    mae_delta = metric_delta(baseline, zero, "mae")

    default_checkpoint = "outputs/checkpoints/unrolled_complex_recon_c3_acc4_best.pt"
    checkpoint = metadata.get("checkpoint", default_checkpoint)
    config_status = metadata.get("config_status", "unknown")
    num_cascades = metadata.get("num_cascades", "3")
    base_channels = metadata.get("base_channels", "8")
    epochs = metadata.get("epochs", "10")
    test_file_count = metadata.get("test_file_count", "")
    train_file_count = metadata.get("train_file_count", "")
    test_digest = metadata.get("test_files_digest", "")

    return f"""# Unrolled Complex U-Net Baseline Report

## Status

The code already contains the Unrolled Complex U-Net baseline. It is implemented
by `UnrolledComplexUNetRecon` in:

```text
src/mri_recon/models/unrolled_frequency_aware.py
```

It is selected with:

```powershell
python scripts\\train_unrolled_frequency_aware_recon_multifile.py --model-type complex
```

This is the most important baseline for the proposed unrolled FA/KAN-style
method because it keeps the same unrolled reconstruction and soft k-space data
consistency structure, but removes the frequency-aware and KAN-style gates.

## Current Configuration

| Field | Value |
| --- | --- |
| checkpoint | `{checkpoint}` |
| config status | `{config_status}` |
| num cascades | `{num_cascades}` |
| base channels | `{base_channels}` |
| epochs | `{epochs}` |
| train files | `{train_file_count}` |
| test files | `{test_file_count}` |
| test split digest | `{test_digest}` |

## Summary Metrics

| Method | PSNR | SSIM | MAE |
| --- | ---: | ---: | ---: |
| zero-filled | {format_summary_row(zero)} |
| unrolled complex U-Net | {format_summary_row(baseline)} |

Delta vs zero-filled:

```text
PSNR: {signed(psnr_delta)} dB
SSIM: {signed(ssim_delta)}
MAE:  {signed(mae_delta, digits=6)}
```

## Per-Slice Behavior

Evaluated paired slices: `{per_slice['slices']}`

```text
PSNR improved: {per_slice['psnr_improved']} / {per_slice['slices']} slices
SSIM improved: {per_slice['ssim_improved']} / {per_slice['slices']} slices
MAE improved:  {per_slice['mae_improved']} / {per_slice['slices']} slices
Mean PSNR delta: {signed(float(per_slice['psnr_delta_mean']))} dB
Mean SSIM delta: {signed(float(per_slice['ssim_delta_mean']))}
Mean MAE delta:  {signed(float(per_slice['mae_delta_mean']), digits=6)}
```

## Interpretation

This baseline is structurally fair, but the current checkpoint is not yet a
strong final baseline: in the existing run, it is slightly worse than
zero-filled on average. That does not make it useless; it is still valuable
because it isolates the effect of replacing a plain complex U-Net regularizer
with the frequency-aware regularizer.

For the final paper table, rerun `complex`, `fa`, and `kan` with the same number
of cascades, epochs, middle-slice margin, split, and mask seed. Then report only
rows marked `matched` by `scripts\\summarize_unrolled_ablation.py`.

## Recommended Final Baseline Command

```powershell
python scripts\\train_unrolled_frequency_aware_recon_multifile.py `
    --model-type complex --epochs 20 --num-cascades 5 --base-channels 8 `
    --seed 42 --mask-seed 42 --middle-slice-margin 5
python scripts\\evaluate_unrolled_frequency_aware_recon.py `
    --checkpoint-path outputs\\checkpoints\\unrolled_complex_recon_c5_acc4_best.pt
python scripts\\summarize_unrolled_ablation.py
```
"""


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    report = build_report(
        metrics_path=Path(args.metrics_path),
        per_slice_path=Path(args.per_slice_path),
        ablation_path=Path(args.ablation_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
