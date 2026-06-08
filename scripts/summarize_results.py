from __future__ import annotations

import csv
from pathlib import Path


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_summary_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "psnr", "ssim", "mae"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    metrics_dir = Path("outputs/metrics")
    output_path = metrics_dir / "final_summary_metrics.csv"

    summary_rows = []

    unet_path = metrics_dir / "unet_vs_zero_filled_metrics.csv"
    complex_path = metrics_dir / "complex_unet_with_dc_metrics.csv"
    naive_dc_path = metrics_dir / "unet_with_dc_metrics.csv"

    if unet_path.exists():
        for row in read_metrics_csv(unet_path):
            if row["method"] == "unet":
                summary_rows.append(
                    {
                        "method": "image_unet",
                        "psnr": row["psnr"],
                        "ssim": row["ssim"],
                        "mae": row["mae"],
                    }
                )
            elif row["method"] == "zero_filled":
                summary_rows.append(row)

    if naive_dc_path.exists():
        for row in read_metrics_csv(naive_dc_path):
            if row["method"] == "unet_dc":
                summary_rows.append(
                    {
                        "method": "naive_image_unet_dc",
                        "psnr": row["psnr"],
                        "ssim": row["ssim"],
                        "mae": row["mae"],
                    }
                )

    if complex_path.exists():
        for row in read_metrics_csv(complex_path):
            if row["method"] in {"complex_unet", "complex_unet_dc"}:
                summary_rows.append(row)

    preferred_order = {
        "zero_filled": 0,
        "image_unet": 1,
        "naive_image_unet_dc": 2,
        "complex_unet": 3,
        "complex_unet_dc": 4,
    }

    summary_rows = sorted(
        summary_rows,
        key=lambda row: preferred_order.get(row["method"], 999),
    )

    write_summary_csv(summary_rows, output_path)

    print("Final summary metrics:")
    for row in summary_rows:
        print(
            f"{row['method']:22s} | "
            f"PSNR={float(row['psnr']):.4f} | "
            f"SSIM={float(row['ssim']):.4f} | "
            f"MAE={float(row['mae']):.6f}"
        )

    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
