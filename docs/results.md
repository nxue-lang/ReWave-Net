# Results

## Current Matched Comparison

The current matched comparison uses:

- fastMRI single-coil knee data;
- a fixed multi-file train/test split;
- acceleration `4` and center fraction `0.08`;
- five cascades and eight base channels;
- 20 training epochs;
- seed and mask seed `42`; and
- target-scale PSNR, SSIM, and MAE conversion.

| Method | PSNR mean | SSIM mean | MAE mean | Delta PSNR vs zero-filled |
| --- | ---: | ---: | ---: | ---: |
| Zero-filled | 25.4182 | 0.5456 | 0.042191 | 0.0000 dB |
| Unrolled Complex U-Net | 26.4522 | 0.5742 | 0.039058 | +1.0340 dB |
| ReWave-Net | **27.0594** | **0.5918** | **0.037323** | **+1.6412 dB** |

The ReWave-Net checkpoint learned the following cascade-wise soft
data-consistency weights from a common initialization of `0.1`:

```text
[0.772, 0.910, 0.910, 0.938, 0.958]
```

## Interpretation

This result shows that the complete ReWave-Net model outperforms the matched
unrolled Complex U-Net baseline on the current fixed split. It does not by
itself isolate the contribution of every component.

The strongest component-level claim requires matched ablations for:

1. ReWave-Net without measured-residual conditioning;
2. ReWave-Net without wavelet routing;
3. ReWave-Net with fixed rather than learned soft-DC weights; and
4. matched frequency-aware and KAN-style historical baselines.

## Reproduction

Train and evaluate ReWave-Net:

```bash
python scripts/train_unrolled_frequency_aware_recon_multifile.py \
  --model-type residual_wavelet \
  --epochs 20 \
  --num-cascades 5 \
  --base-channels 8 \
  --seed 42 \
  --mask-seed 42

python scripts/evaluate_unrolled_frequency_aware_recon.py \
  --checkpoint-path outputs/checkpoints/unrolled_residual_wavelet_recon_c5_acc4_best.pt
```

Summarize available matched models:

```bash
python scripts/summarize_unrolled_ablation.py
```

Generated checkpoints and metrics remain local under `outputs/` and are not
committed to the source repository.
