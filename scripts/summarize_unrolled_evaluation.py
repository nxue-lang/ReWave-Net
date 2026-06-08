from __future__ import annotations

import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    summary_path = Path("outputs/metrics/unrolled_frequency_aware_recon_metrics.csv")
    per_slice_path = Path(
        "outputs/metrics/unrolled_frequency_aware_recon_per_slice_metrics.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary metrics: {summary_path}")
    if not per_slice_path.exists():
        raise FileNotFoundError(f"Missing per-slice metrics: {per_slice_path}")

    summary_rows = read_rows(summary_path)
    per_slice_rows = read_rows(per_slice_path)

    paired: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in per_slice_rows:
        key = (row["file_path"], row["slice_index"])
        paired.setdefault(key, {})[row["method"]] = row

    pairs = [
        methods
        for methods in paired.values()
        if "zero_filled" in methods and "unrolled_frequency_aware" in methods
    ]

    psnr_improved = 0
    ssim_improved = 0
    mae_improved = 0

    for methods in pairs:
        zero = methods["zero_filled"]
        unrolled = methods["unrolled_frequency_aware"]

        if float(unrolled["psnr"]) > float(zero["psnr"]):
            psnr_improved += 1
        if float(unrolled["ssim"]) > float(zero["ssim"]):
            ssim_improved += 1
        if float(unrolled["mae"]) < float(zero["mae"]):
            mae_improved += 1

    by_method = {row["method"]: row for row in summary_rows}
    zero = by_method["zero_filled"]
    unrolled = by_method["unrolled_frequency_aware"]

    psnr_gain = float(unrolled["psnr_mean"]) - float(zero["psnr_mean"])
    ssim_gain = float(unrolled["ssim_mean"]) - float(zero["ssim_mean"])
    mae_gain = float(unrolled["mae_mean"]) - float(zero["mae_mean"])

    print("Unrolled KAN-style frequency-aware reconstruction summary")
    print(f"Evaluated slices: {len(pairs)}")
    print(
        f"PSNR: {float(zero['psnr_mean']):.4f} -> "
        f"{float(unrolled['psnr_mean']):.4f} "
        f"({psnr_gain:+.4f} dB)"
    )
    print(
        f"SSIM: {float(zero['ssim_mean']):.4f} -> "
        f"{float(unrolled['ssim_mean']):.4f} "
        f"({ssim_gain:+.4f})"
    )
    print(
        f"MAE: {float(zero['mae_mean']):.6f} -> "
        f"{float(unrolled['mae_mean']):.6f} "
        f"({mae_gain:+.6f})"
    )
    print(f"PSNR improved slices: {psnr_improved}/{len(pairs)}")
    print(f"SSIM improved slices: {ssim_improved}/{len(pairs)}")
    print(f"MAE improved slices: {mae_improved}/{len(pairs)}")


if __name__ == "__main__":
    main()
