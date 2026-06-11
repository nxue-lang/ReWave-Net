# ReWave-Net Results

This directory contains lightweight, version-controlled artifacts for the
matched ReWave-Net experiment reported in the repository README.

## Experiment

| Setting | Value |
| --- | --- |
| Dataset | fastMRI single-coil knee |
| Train/held-out volumes | 159 / 40 |
| Train split hash | `c3f49347692ef2806afb3db6` |
| Held-out split hash | `e281eb8779521500f31e5870` |
| Acceleration | 4 |
| Center fraction | 0.08 |
| Cascades | 5 |
| Base channels | 8 |
| Shared regularizer | Yes |
| Initial soft-DC weight | 0.1 |
| Matched comparison epochs | 20 |
| Seed / mask seed | 42 / 42 |

Split hashes are 12-byte BLAKE2s digests of sorted fastMRI volume filenames
joined by newline characters without a trailing newline. Raw data paths and
files are not published.

## Matched Comparison

| Method | PSNR mean/std | SSIM mean/std | MAE mean/std |
| --- | --- | --- | --- |
| Zero-filled | 25.4182 / 2.2762 | 0.5456 / 0.0932 | 0.042191 / 0.011912 |
| Unrolled Complex U-Net | 26.4522 / 2.8305 | 0.5742 / 0.1047 | 0.039058 / 0.013241 |
| ReWave-Net | **27.0594 / 3.1976** | **0.5918 / 0.1111** | **0.037323 / 0.013716** |

All three rows use the same held-out split and metric conversion. ReWave-Net
and the Complex U-Net baseline use matched training configurations. Metrics
are computed per slice after a `320 x 320` center crop and target-maximum
scaling, then averaged across slices.

## Extended Training Result

The `v0.1.1` ReWave-Net checkpoint continues training to 40 total epochs and
selects epoch 39:

| Method | PSNR mean/std | SSIM mean/std | MAE mean/std |
| --- | --- | --- | --- |
| ReWave-Net `v0.1.1` | **27.1215 / 3.2338** | **0.5943 / 0.1121** | **0.037148 / 0.013750** |

The 20-epoch Complex U-Net baseline was not retrained to 40 epochs, so this
extended-training result is not presented as a matched-epoch baseline
comparison.

## Files

- `experiment_config.json`: compact experiment and checkpoint metadata.
- `rewave_net_example.png`: representative target, zero-filled reconstruction,
  ReWave-Net reconstruction, and error map.

![ReWave-Net reconstruction example](rewave_net_example.png)

## Pretrained Checkpoint

The best checkpoint is attached to the
[GitHub `v0.1.1` release](https://github.com/nxue-lang/ReWave-Net/releases/tag/v0.1.1)
as:

```text
rewave_c5_acc4_best.pt
```

SHA256:

```text
fcc5e92cdef9325f306b8c95fb1318ab1b55dca7aef5c3d6469fabc0611fe043
```

Training continued from the original 20-epoch checkpoint to 40 total epochs.
The checkpoint was selected at epoch 39 with validation complex L1 loss
`0.0296036175`. Its learned soft-DC weights are approximately:

```text
[0.9545, 0.9954, 0.9969, 0.9970, 0.9878]
```

The same 40 held-out volumes were used for checkpoint selection and the
reported metrics. These results are therefore validation/evaluation results,
not an independent test-set estimate.
