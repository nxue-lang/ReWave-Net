# Residual-Conditioned Wavelet Unrolled Reconstruction

## Core idea

The new model uses measured k-space errors to control how strongly each
unrolled cascade processes anatomical structure and local detail.

At the start of cascade `t`, it compares the current complex reconstruction
`x_(t-1)` with the acquired k-space `y` at sampled locations:

```text
r_t = M * (y - F x_(t-1))
```

The sampled residual is summarized in low, mid, and high radial k-space bands.
Each band uses a sample-count-normalized relative error, followed by `log1p`
compression for stable conditioning.

These three residual values and the normalized cascade index condition every
wavelet routing block in the complex U-Net regularizer.

## Wavelet routing block

Each block applies an orthonormal 2D Haar transform:

```text
feature map
  -> LL: low-frequency structure
  -> LH / HL / HH: directional detail bands
```

The `LL` band uses a large-kernel depthwise convolution for structural context.
The three detail bands use a residual-style convolutional branch. A channel
gate receives:

```text
low feature summary
high feature summary
low / mid / high measured k-space residuals
cascade progress
```

It routes the processed Haar bands before the inverse Haar transform. The block
then adds the routed result back to its input.

## Cascade

```text
current complex image
  -> normalized measured band residuals
  -> residual-conditioned wavelet complex U-Net
  -> candidate complex image
  -> learnable soft k-space data consistency
  -> next complex image
```

The residual is computed before the regularizer in each cascade. This avoids a
circular same-pass dependency while ensuring every stage reacts to the latest
measured-data mismatch.

## Commands

Smoke test:

```powershell
python scripts/test_rewave_net.py
```

Small training run:

```powershell
python scripts/train_rewave_net.py --model-type rewave --epochs 1 --num-cascades 2 --base-channels 4 --max-train-files 2 --max-test-files 1 --max-train-samples 8 --max-test-samples 4 --disable-progress
```

Full matched experiment:

```powershell
python scripts/train_rewave_net.py --model-type rewave --epochs 20 --num-cascades 5 --base-channels 8 --seed 42 --mask-seed 42
python scripts/evaluate_rewave_net.py --checkpoint-path outputs/checkpoints/rewave_c5_acc4_best.pt
```

## Required ablation

Use identical data splits, masks, cascades, base channels, epochs, and metric
conversion for:

```text
Unrolled Complex U-Net
ReWave-Net
```

The new mechanism should only be claimed as beneficial if it improves the
matched unrolled Complex U-Net baseline.
