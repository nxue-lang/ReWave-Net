# Results

## Current Matched Comparison

The current matched comparison uses:

- fastMRI single-coil knee data;
- a fixed 159/40-volume train/held-out split;
- acceleration `4` and center fraction `0.08`;
- five cascades and eight base channels;
- 20 training epochs;
- seed and mask seed `42`; and
- per-slice target-scale PSNR, SSIM, and MAE conversion.

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

## Extended Training Checkpoint

The `v0.1.1` release continues ReWave-Net training from 20 to 40 total epochs.
Its best checkpoint occurs at epoch 39:

| Method | Training epochs | PSNR mean | SSIM mean | MAE mean |
| --- | ---: | ---: | ---: | ---: |
| ReWave-Net `v0.1.0` | 20 | 27.0594 | 0.5918 | 0.037323 |
| ReWave-Net `v0.1.1` | 40 total, best at 39 | **27.1215** | **0.5943** | **0.037148** |

The extended checkpoint has validation complex L1 loss `0.0296036175` and
learned soft-DC weights approximately:

```text
[0.955, 0.995, 0.997, 0.997, 0.988]
```

The Complex U-Net baseline has not been retrained to 40 epochs. Therefore, the
`v0.1.1` result is reported as an improved ReWave-Net checkpoint, not as a
matched-epoch comparison against that baseline.

## Evaluation Protocol

The training script evaluates complex L1 loss on the 40 held-out volumes after
each epoch and selects the best checkpoint using that loss. The reported
PSNR, SSIM, and MAE values are then computed on the same held-out volumes.
They should therefore be interpreted as validation/evaluation results, not as
an independent test-set estimate.

For each evaluated slice, the code:

1. converts the two real/imaginary channels to a complex magnitude image;
2. center-crops the prediction and target to `320 x 320`;
3. divides both by that slice's target-magnitude maximum;
4. computes PSNR and SSIM with `data_range=1.0`, plus MAE; and
5. reports the arithmetic mean and standard deviation across slices.

Up to the middle 11 slices of each volume are used
(`middle_slice_margin=5`). This is the repository's matched internal
evaluation protocol and is not the official fastMRI leaderboard protocol.

## Interpretation

This result shows that the complete ReWave-Net model outperforms the matched
unrolled Complex U-Net baseline on the current fixed held-out split. It does
not by itself isolate the contribution of every component or estimate
performance on an independent test set.

The strongest component-level claim requires matched ablations for:

1. ReWave-Net without measured-residual conditioning;
2. ReWave-Net without wavelet routing;
3. ReWave-Net with fixed rather than learned soft-DC weights; and
4. additional component-level ablations when making stronger causal claims.

## Reproduction

Train and evaluate ReWave-Net:

```bash
python scripts/train_rewave_net.py \
  --model-type rewave \
  --epochs 20 \
  --num-cascades 5 \
  --base-channels 8 \
  --seed 42 \
  --mask-seed 42

python scripts/evaluate_rewave_net.py \
  --checkpoint-path outputs/checkpoints/rewave_c5_acc4_best.pt
```

Summarize available matched models:

```bash
python scripts/summarize_matched_comparison.py
```

Generated checkpoints and metrics remain local under `outputs/` and are not
committed to the source repository.
