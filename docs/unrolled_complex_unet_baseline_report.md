# Unrolled Complex U-Net Baseline Report

## Status

The code already contains the Unrolled Complex U-Net baseline. It is implemented
by `UnrolledComplexUNetRecon` in:

```text
src/mri_recon/models/unrolled_frequency_aware.py
```

It is selected with:

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type complex
```

This is the most important baseline for the proposed unrolled FA/KAN-style
method because it keeps the same unrolled reconstruction and soft k-space data
consistency structure, but removes the frequency-aware and KAN-style gates.

## Current Configuration

| Field | Value |
| --- | --- |
| checkpoint | `outputs/checkpoints/unrolled_complex_recon_c3_acc4_best.pt` |
| config status | `matched` |
| num cascades | `3` |
| base channels | `8` |
| epochs | `10` |
| train files | `159` |
| test files | `40` |
| test split digest | `fcfa2f633bff` |

## Summary Metrics

| Method | PSNR | SSIM | MAE |
| --- | ---: | ---: | ---: |
| zero-filled | 25.3412 +/- 2.1944 | 0.5448 +/- 0.0893 | 0.042431 +/- 0.011451 |
| unrolled complex U-Net | 25.1662 +/- 2.2858 | 0.5412 +/- 0.0920 | 0.043516 +/- 0.013108 |

Delta vs zero-filled:

```text
PSNR: -0.1750 dB
SSIM: -0.0037
MAE:  +0.001086
```

## Per-Slice Behavior

Evaluated paired slices: `280`

```text
PSNR improved: 63 / 280 slices
SSIM improved: 63 / 280 slices
MAE improved:  58 / 280 slices
Mean PSNR delta: -0.1750 dB
Mean SSIM delta: -0.0037
Mean MAE delta:  +0.001086
```

## Interpretation

This baseline is structurally fair, but the current checkpoint is not yet a
strong final baseline: in the existing run, it is slightly worse than
zero-filled on average. That does not make it useless; it is still valuable
because it isolates the effect of replacing a plain complex U-Net regularizer
with the frequency-aware regularizer.

For the final paper table, rerun `complex`, `fa`, and `kan` with the same number
of cascades, epochs, middle-slice margin, split, and mask seed. Then report only
rows marked `matched` by `scripts\summarize_unrolled_ablation.py`.

## Recommended Final Baseline Command

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py `
    --model-type complex --epochs 20 --num-cascades 5 --base-channels 8 `
    --seed 42 --mask-seed 42 --middle-slice-margin 5
python scripts\evaluate_unrolled_frequency_aware_recon.py `
    --checkpoint-path outputs\checkpoints\unrolled_complex_recon_c5_acc4_best.pt
python scripts\summarize_unrolled_ablation.py
```
