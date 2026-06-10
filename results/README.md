# ReWave-Net Results

This directory contains lightweight, version-controlled artifacts for the
matched ReWave-Net experiment reported in the repository README.

## Experiment

| Setting | Value |
| --- | --- |
| Dataset | fastMRI single-coil knee |
| Train/test volumes | 159 / 40 |
| Train split hash | `c3f49347692ef2806afb3db6` |
| Test split hash | `e281eb8779521500f31e5870` |
| Acceleration | 4 |
| Center fraction | 0.08 |
| Cascades | 5 |
| Base channels | 8 |
| Shared regularizer | Yes |
| Initial soft-DC weight | 0.1 |
| Epochs | 20 |
| Seed / mask seed | 42 / 42 |

Split hashes are BLAKE2s digests of sorted fastMRI volume filenames. Raw data
paths and files are not published.

## Matched Comparison

| Method | PSNR mean/std | SSIM mean/std | MAE mean/std |
| --- | --- | --- | --- |
| Zero-filled | 25.4182 / 2.2762 | 0.5456 / 0.0932 | 0.042191 / 0.011912 |
| Unrolled Complex U-Net | 26.4522 / 2.8305 | 0.5742 / 0.1047 | 0.039058 / 0.013241 |
| ReWave-Net | **27.0594 / 3.1976** | **0.5918 / 0.1111** | **0.037323 / 0.013716** |

All three rows use the same test split and metric conversion. ReWave-Net and
the Complex U-Net baseline use matched training configurations.

## Files

- `experiment_config.json`: compact experiment and checkpoint metadata.
- `rewave_net_example.png`: representative target, zero-filled reconstruction,
  ReWave-Net reconstruction, and error map.

![ReWave-Net reconstruction example](rewave_net_example.png)

## Pretrained Checkpoint

The best checkpoint is attached to the GitHub `v0.1.0` release as:

```text
rewave_c5_acc4_best.pt
```

SHA256:

```text
5d0b7b523e0d220f66e5f965a2e742c48297b058c0826d055c44376c0ebb7f05
```

The checkpoint was selected at epoch 20 with validation complex L1 loss
`0.0296702308`. Its learned soft-DC weights are approximately:

```text
[0.7720, 0.9104, 0.9096, 0.9380, 0.9584]
```
