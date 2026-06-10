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

- $x_{t-1}$: complex image entering cascade $t$
- $y$: undersampled measured k-space
- $M$: binary sampling mask, expanded over k-space height
- $\mathcal{F}$ and $\mathcal{F}^{-1}$: centered orthonormal FFT and inverse FFT
- $T$: total number of cascades
- $B_{\mathrm{low}}$, $B_{\mathrm{mid}}$, $B_{\mathrm{high}}$: radial k-space band masks

The implementation stores complex images and k-space as two real/imaginary
channels.

## 1. Measured Band-Residual Controller

At the start of cascade $t$, the predicted k-space is:

$$
k_{\mathrm{pred},t} = \mathcal{F}x_{t-1}.
$$

Residual statistics are computed only where k-space was acquired. For band
$b \in \{\mathrm{low}, \mathrm{mid}, \mathrm{high}\}$, define the sampled band:

$$
S_b = M \cap B_b.
$$

ReWave-Net defines its conditioning statistic as:

$$
e_{b,t}
=
\log\left(
1+
\frac{
\frac{1}{|S_b|}\sum_{u\in S_b}
\left|y(u)-k_{\mathrm{pred},t}(u)\right|
}{
\frac{1}{|S_b|}\sum_{u\in S_b}|y(u)|+\varepsilon
}
\right).
$$

The radial bands use normalized-radius boundaries:

$$
\begin{aligned}
B_{\mathrm{low}}  &= \{u : \rho(u) < 1/3\}, \\
B_{\mathrm{mid}}  &= \{u : 1/3 \le \rho(u) < 2/3\}, \\
B_{\mathrm{high}} &= \{u : \rho(u) \ge 2/3\}.
\end{aligned}
$$

The sample-count normalization prevents bands with more acquired locations
from automatically producing larger residual values. Measurement-magnitude
normalization makes the statistic relative to the measured signal scale, and
`log1p` compresses large values.

Using zero-based cascade index $t$, the cascade progress is:

$$
p_t = \frac{t}{\max(T-1,\,1)}.
$$

The condition vector supplied to every wavelet-routing block is:

$$
c_t =
\left[
e_{\mathrm{low},t},
e_{\mathrm{mid},t},
e_{\mathrm{high},t},
p_t
\right].
$$

These residual definitions and their use as routing conditions are ReWave-Net
design choices.

## 2. Residual-Conditioned Wavelet Routing

Each conditioned convolution block first extracts features and then applies an
orthonormal 2D Haar transform:

$$
(X_{LL}, X_{LH}, X_{HL}, X_{HH}) = \operatorname{DWT}(X).
$$

$X_{LL}$ is processed by a large-kernel structure branch. $X_{LH}$, $X_{HL}$, and $X_{HH}$
share a detail branch:

$$
\begin{aligned}
L &= f_{\mathrm{low}}(X_{LL}), \\
H_q &= f_{\mathrm{high}}(X_q),
\qquad q \in \{LH, HL, HH\}.
\end{aligned}
$$

The block summarizes the processed branches:

$$
\bar{l} = \operatorname{GAP}(L),
\qquad
\bar{h} =
\frac{1}{3}\sum_{q\in\{LH,HL,HH\}}
\operatorname{GAP}(|H_q|).
$$

It then predicts one routing value per sample and feature channel:

$$
g = \sigma\!\left(
\operatorname{MLP}\!\left([\bar{l},\bar{h},c_t]\right)
\right).
$$

The routed subbands are:

$$
L_{\mathrm{routed}} = (1-g)\odot L,
\qquad
H_{q,\mathrm{routed}} = g\odot H_q.
$$

and the routed feature map is reconstructed with the inverse Haar transform:

$$
z =
\operatorname{IWT}\!\left(
L_{\mathrm{routed}},
H_{LH,\mathrm{routed}},
H_{HL,\mathrm{routed}},
H_{HH,\mathrm{routed}}
\right).
$$

When $g$ is closer to zero, that feature channel retains more processed $LL$
structure. When $g$ is closer to one, it retains more processed detail-band
features.

The three measured residuals jointly condition the MLP. The implementation
does not force a one-to-one mapping between a k-space residual band and a Haar
subband, and the gate is channel-wise rather than spatially varying.

## 3. Cascade-Wise Soft Data Consistency

After the wavelet U-Net produces candidate reconstruction $\widetilde{x}_t$,
ReWave-Net applies weighted k-space data consistency:

$$
\begin{aligned}
k_t
&=
\mathcal{F}\widetilde{x}_t
+
\lambda_t M\odot
\left(y-\mathcal{F}\widetilde{x}_t\right), \\
x_t &= \mathcal{F}^{-1}k_t.
\end{aligned}
$$

Each cascade has an independently learned scalar:

$$
\lambda_t = \sigma(\alpha_t),
\qquad
\lambda_t \in [0,1].
$$

$\lambda_t=0$ keeps the network prediction unchanged at measured locations,
while $\lambda_t=1$ replaces those locations with the measurements. Weighted
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
