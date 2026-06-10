from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mri_recon.data.multi_file_complex_dataset import (
    MultiFileFastMRIComplexSingleCoilDataset,
)
from mri_recon.models.unrolled_frequency_aware import (
    UnrolledComplexUNetRecon,
    UnrolledFrequencyAwareRecon,
    UnrolledKANFrequencyAwareRecon,
    UnrolledResidualConditionedWaveletRecon,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an unrolled frequency-aware MRI reconstruction model."
    )

    parser.add_argument("--data-dir", type=str, default="data/knee_singlecoil_val")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["complex", "fa", "kan", "residual_wavelet"],
        default="residual_wavelet",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--num-cascades", type=int, default=5)
    parser.add_argument("--unshared-denoiser", action="store_true")
    parser.add_argument("--initial-dc-weight", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--middle-slice-margin", type=int, default=5)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume training state from a checkpoint.",
    )
    parser.add_argument(
        "--fast-profile",
        action="store_true",
        help=(
            "Use a faster practical profile: batch size 2, four data workers, "
            "validation every five epochs, and fewer middle slices."
        ),
    )

    args = parser.parse_args()
    if args.fast_profile:
        args.batch_size = max(args.batch_size, 2)
        args.num_workers = max(args.num_workers, 4)
        args.val_every = max(args.val_every, 5)
        args.middle_slice_margin = min(args.middle_slice_margin, 3)
    if args.val_every < 1:
        raise ValueError(f"--val-every must be >= 1, got {args.val_every}")
    if args.mask_seed is None:
        args.mask_seed = args.seed

    return args


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def split_files(
    h5_paths: list[Path],
    train_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    paths = list(h5_paths)
    rng.shuffle(paths)

    split_index = int(len(paths) * train_ratio)
    return paths[:split_index], paths[split_index:]


def save_file_split(
    train_paths: list[Path],
    test_paths: list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("split,file_path\n")
        for path in train_paths:
            file.write(f"train,{path.as_posix()}\n")
        for path in test_paths:
            file.write(f"test,{path.as_posix()}\n")


def build_model(args: argparse.Namespace) -> nn.Module:
    shared_denoiser = not args.unshared_denoiser
    model_kwargs = {
        "num_cascades": args.num_cascades,
        "base_channels": args.base_channels,
        "shared_denoiser": shared_denoiser,
        "initial_dc_weight": args.initial_dc_weight,
    }

    if args.model_type == "complex":
        return UnrolledComplexUNetRecon(**model_kwargs)
    if args.model_type == "fa":
        return UnrolledFrequencyAwareRecon(**model_kwargs)
    if args.model_type == "residual_wavelet":
        return UnrolledResidualConditionedWaveletRecon(**model_kwargs)

    return UnrolledKANFrequencyAwareRecon(**model_kwargs)


def save_training_metrics(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "train_loss", "val_loss", "dc_weights"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curve(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Complex L1 Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    show_progress: bool = False,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Validation", leave=False)

        for batch in iterator:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target_complex"].to(device, non_blocking=True)
            measured_kspaces = batch["measured_kspace"].to(
                device, non_blocking=True
            )
            masks = batch["mask"].to(device, non_blocking=True)

            predictions = model(
                image=inputs,
                measured_kspace=measured_kspaces,
                mask=masks,
            )
            loss = loss_fn(predictions, targets)
            losses.append(float(loss.item()))

    return float(np.mean(losses))


def format_dc_weights(model: nn.Module) -> str:
    weights = model.dc_weights.detach().cpu().numpy().tolist()
    return ";".join(f"{weight:.4f}" for weight in weights)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    splits_dir = output_dir / "splits"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Using AMP: {use_amp}")

    data_dir = Path(args.data_dir)
    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No h5 files found in {data_dir}")

    train_paths, test_paths = split_files(
        h5_paths=h5_paths,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    if args.max_train_files is not None:
        train_paths = train_paths[: args.max_train_files]
    if args.max_test_files is not None:
        test_paths = test_paths[: args.max_test_files]

    split_path = splits_dir / f"unrolled_multifile_split_seed{args.seed}.csv"
    save_file_split(
        train_paths=train_paths, test_paths=test_paths, output_path=split_path
    )

    print(f"Total files: {len(h5_paths)}")
    print(f"Train files: {len(train_paths)}")
    print(f"Test files: {len(test_paths)}")
    print(f"Saved split to: {split_path}")

    train_dataset = MultiFileFastMRIComplexSingleCoilDataset(
        h5_paths=train_paths,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        use_middle_slices_only=True,
        middle_slice_margin=args.middle_slice_margin,
        mask_seed=args.mask_seed,
    )
    test_dataset = MultiFileFastMRIComplexSingleCoilDataset(
        h5_paths=test_paths,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        use_middle_slices_only=True,
        middle_slice_margin=args.middle_slice_margin,
        mask_seed=args.mask_seed,
    )

    if args.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(args.max_train_samples, len(train_dataset))),
        )
    if args.max_test_samples is not None:
        test_dataset = Subset(
            test_dataset,
            range(min(args.max_test_samples, len(test_dataset))),
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_name = (
        f"unrolled_{args.model_type}_recon_c{args.num_cascades}_"
        f"acc{args.acceleration}_best.pt"
    )
    last_checkpoint_name = (
        f"unrolled_{args.model_type}_recon_c{args.num_cascades}_"
        f"acc{args.acceleration}_last.pt"
    )
    best_val_loss = float("inf")
    history = []
    start_epoch = 1

    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

        completed_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        resumed_best_val_loss = float(
            resume_checkpoint.get(
                "best_val_loss",
                resume_checkpoint.get("val_loss", float("inf")),
            )
        )
        best_val_loss = (
            resumed_best_val_loss
            if np.isfinite(resumed_best_val_loss)
            else float("inf")
        )
        print(f"Resuming from: {resume_path}")
        print(f"Starting at epoch: {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []

        iterator = train_loader
        if not args.disable_progress:
            iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d}/{args.epochs:03d}",
                leave=False,
            )

        for batch in iterator:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target_complex"].to(device, non_blocking=True)
            measured_kspaces = batch["measured_kspace"].to(
                device, non_blocking=True
            )
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(
                    image=inputs,
                    measured_kspace=measured_kspaces,
                    mask=masks,
                )
                loss = loss_fn(predictions, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.item()))
            if not args.disable_progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(train_losses))
        should_validate = (
            epoch == 1 or epoch == args.epochs or epoch % args.val_every == 0
        )
        val_loss = float("nan")
        if should_validate:
            val_loss = evaluate_loss(
                model=model,
                dataloader=test_loader,
                loss_fn=loss_fn,
                device=device,
                show_progress=not args.disable_progress,
            )
        dc_weights = format_dc_weights(model)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dc_weights": dc_weights,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"dc_weights={dc_weights}"
        )

        last_checkpoint_path = checkpoints_dir / last_checkpoint_name
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "train_files": [path.as_posix() for path in train_paths],
                "test_files": [path.as_posix() for path in test_paths],
            },
            last_checkpoint_path,
        )

        if should_validate and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoints_dir / checkpoint_name
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                    "train_files": [path.as_posix() for path in train_paths],
                    "test_files": [path.as_posix() for path in test_paths],
                },
                checkpoint_path,
            )

    metrics_path = (
        metrics_dir
        / f"unrolled_{args.model_type}_recon_c{args.num_cascades}_training_metrics.csv"
    )
    save_training_metrics(history, metrics_path)

    loss_curve_path = (
        figures_dir
        / f"unrolled_{args.model_type}_recon_c{args.num_cascades}_loss_curve.png"
    )
    plot_loss_curve(history, loss_curve_path)

    print("Unrolled frequency-aware training completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved checkpoint to: {checkpoints_dir / checkpoint_name}")
    print(f"Saved last checkpoint to: {checkpoints_dir / last_checkpoint_name}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved loss curve to: {loss_curve_path}")


if __name__ == "__main__":
    main()
