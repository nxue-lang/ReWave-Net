# ReWave-Net Method

## Overview

ReWave-Net is a single-coil complex MRI reconstruction model. It repeats the
following sequence for each cascade:

```text
current complex reconstruction
  -> measured low/mid/high k-space residual controller
  -> residual-conditioned wavelet complex U-Net
  -> candidate complex reconstruction
  -> learned soft k-space data consistency
  -> next cascade
```

By default, regularizer weights are shared across cascades. The measured
residual condition still changes with each sample and cascade, allowing the
same regularizer to change its structure/detail routing according to the
current measured-data mismatch. The training CLI also supports unshared
regularizers with `--unshared-denoiser`.

## Notation

- `x_(t-1)`: complex image entering cascade `t`
- `y`: undersampled measured k-space
- `M`: binary sampling mask, expanded over k-space height
- `F` and `F^-1`: centered orthonormal FFT and inverse FFT
- `T`: total number of cascades
- `B_low`, `B_mid`, `B_high`: radial k-space band masks

The implementation stores complex images and k-space as two real/imaginary
channels.

## 1. Measured Band-Residual Controller

At the start of cascade `t`, the predicted k-space is:

```text
k_pred,t = F x_(t-1)
```

Residual statistics are computed only where k-space was acquired. For band
`b` in `{low, mid, high}`, define the sampled band:

```text
S_b = M intersect B_b
```

ReWave-Net defines its conditioning statistic as:

```text
e_b,t = log(1 + mean_(u in S_b) |y(u) - k_pred,t(u)|
                  / (mean_(u in S_b) |y(u)| + epsilon))
```

The radial bands use normalized-radius boundaries:

```text
B_low:  radius < 1/3
B_mid:  1/3 <= radius < 2/3
B_high: radius >= 2/3
```

The sample-count normalization prevents bands with more acquired locations
from automatically producing larger residual values. Measurement-magnitude
normalization makes the statistic relative to the measured signal scale, and
`log1p` compresses large values.

Using zero-based cascade index `t`, the cascade progress is:

```text
p_t = t / max(T - 1, 1)
```

The condition vector supplied to every wavelet-routing block is:

```text
c_t = [e_low,t, e_mid,t, e_high,t, p_t]
```

These residual definitions and their use as routing conditions are ReWave-Net
design choices.

## 2. Residual-Conditioned Wavelet Routing

Each conditioned convolution block first extracts features and then applies an
orthonormal 2D Haar transform:

```text
(LL, LH, HL, HH) = DWT(feature map)
```

`LL` is processed by a large-kernel structure branch. `LH`, `HL`, and `HH`
share a detail branch:

```text
L = f_low(LL)
H_q = f_high(q), q in {LH, HL, HH}
```

The block summarizes the processed branches:

```text
l_bar = GAP(L)
h_bar = average_q GAP(|H_q|)
```

It then predicts one routing value per sample and feature channel:

```text
g = sigmoid(MLP([l_bar, h_bar, c_t]))
```

The routed subbands are:

```text
L_routed   = (1 - g) * L
H_q,routed = g * H_q
```

and the routed feature map is reconstructed with the inverse Haar transform:

```text
z = IWT(L_routed, H_LH,routed, H_HL,routed, H_HH,routed)
```

When `g` is closer to zero, that feature channel retains more processed `LL`
structure. When `g` is closer to one, it retains more processed detail-band
features.

The three measured residuals jointly condition the MLP. The implementation
does not force a one-to-one mapping between a k-space residual band and a Haar
subband, and the gate is channel-wise rather than spatially varying.

## 3. Cascade-Wise Soft Data Consistency

After the wavelet U-Net produces candidate reconstruction `x_tilde_t`,
ReWave-Net applies weighted k-space data consistency:

```text
k_t = F x_tilde_t + lambda_t * M * (y - F x_tilde_t)
x_t = F^-1 k_t
```

Each cascade has an independently learned scalar:

```text
lambda_t = sigmoid(alpha_t), lambda_t in [0, 1]
```

`lambda_t = 0` keeps the network prediction unchanged at measured locations,
while `lambda_t = 1` replaces those locations with the measurements. Weighted
data consistency and unrolled cascades are standard reconstruction ideas; the
cascade-wise weights are part of the complete ReWave-Net design but are not
presented as the main methodological contribution.

## Contribution Boundary

ReWave-Net does **not** claim the following individual components as new:

- U-Net-style encoder-decoder regularization;
- Haar DWT/IWT;
- global average pooling and MLP channel gating;
- unrolled cascade reconstruction; or
- weighted soft data consistency.

The proposed mechanism is the repeated control path:

```text
current measured k-space mismatch
  -> sample-normalized low/mid/high residual condition
  -> wavelet structure/detail routing at every U-Net block
  -> candidate reconstruction and learned soft DC
  -> recompute the condition at the next cascade
```

This contribution should be evaluated against a matched unrolled Complex U-Net
and with component-level ablations.

## Source Map

- Band residuals and soft DC:
  [`torch_ops.py`](../src/mri_recon/reconstruction/torch_ops.py)
- Haar transform and residual-conditioned routing:
  [`residual_conditioned_wavelet_unet.py`](../src/mri_recon/models/residual_conditioned_wavelet_unet.py)
- Cascade control and learned DC weights:
  [`rewave_net.py`](../src/mri_recon/models/rewave_net.py)

## Commands

Smoke test:

```bash
python scripts/test_rewave_net.py
```

Full matched ReWave-Net experiment:

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
