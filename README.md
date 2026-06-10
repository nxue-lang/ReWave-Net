# ReWave-Net

ReWave-Net is a measurement-residual-conditioned wavelet unrolled network for
accelerated single-coil MRI reconstruction. At every cascade, it:

1. measures the current prediction error only at acquired k-space locations;
2. summarizes that error in low-, mid-, and high-frequency radial bands;
3. conditions channel-wise Haar wavelet routing on those residuals and the
   cascade index; and
4. applies a learnable per-cascade soft data-consistency update.

![ReWave-Net architecture](docs/neural_network.png)

## Method

For cascade `t`, ReWave-Net computes measured residual statistics from the
current complex reconstruction `x_(t-1)`:

```text
r_t = M * (y - F x_(t-1))
```

The low-, mid-, and high-band residual summaries, together with normalized
cascade progress, condition every wavelet-routing block in a shared complex
U-Net regularizer. The candidate reconstruction is then updated using:

```text
k_t = F x_tilde_t + lambda_t * M * (y - F x_tilde_t)
```

where each `lambda_t` is independently learned and constrained to `[0, 1]`.
See [the method description](docs/residual_conditioned_wavelet_method.md) for
the implementation-level details.

## Results

The current matched experiment uses the same multi-file split, sampling rule,
number of cascades, base channels, epochs, and metric conversion for the
zero-filled, unrolled Complex U-Net, and ReWave-Net comparisons.

| Method | PSNR | SSIM | MAE |
| --- | ---: | ---: | ---: |
| Zero-filled | 25.4182 | 0.5456 | 0.042191 |
| Unrolled Complex U-Net | 26.4522 | 0.5742 | 0.039058 |
| ReWave-Net | **27.0594** | **0.5918** | **0.037323** |

ReWave-Net improves PSNR by `0.6072 dB` over the matched unrolled Complex
U-Net baseline. These are research results on the current fixed split, not a
clinical validation claim. See [the results notes](docs/results.md) for
experiment details and remaining ablations.

## Repository Layout

```text
data/                  Local fastMRI data location; data files are ignored
docs/                  Method, results, and experiment documentation
scripts/               Training, evaluation, baseline, and smoke-test scripts
src/mri_recon/         Reusable models, datasets, metrics, transforms, and DC
outputs/               Local checkpoints, figures, metrics, and splits; ignored
```

The main implementation is:

- `src/mri_recon/models/residual_conditioned_wavelet_unet.py`
- `src/mri_recon/models/rewave_net.py`
- `src/mri_recon/reconstruction/torch_ops.py`

The repository keeps the matched unrolled Complex U-Net as the primary
baseline. Earlier exploratory models remain available in Git history.

## Installation

Python 3.9 or newer is required.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

PyTorch installation can depend on the local CUDA version. If necessary,
install the appropriate PyTorch build first, then install the remaining
requirements.

## Data

Download the fastMRI single-coil knee dataset under its applicable access
terms and place the HDF5 files in:

```text
data/knee_singlecoil_val/
```

Data files and generated outputs are intentionally excluded from Git. See
[data/README.md](data/README.md) for the expected layout.

## Quick Start

Run the model smoke test without downloading the dataset:

```bash
python scripts/test_rewave_net.py
```

Run a small end-to-end training check after placing the data:

```bash
python scripts/train_rewave_net.py \
  --model-type rewave \
  --epochs 1 \
  --num-cascades 2 \
  --base-channels 4 \
  --max-train-files 2 \
  --max-test-files 1 \
  --max-train-samples 8 \
  --max-test-samples 4 \
  --disable-progress
```

Run the full matched ReWave-Net experiment:

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

The complete public script inventory is documented in
[scripts/README.md](scripts/README.md).

## Reproducibility

- Use the same `--seed` and `--mask-seed` for matched comparisons.
- Keep the file split, mask rule, cascades, channels, epochs, and metric
  conversion fixed across models.
- Checkpoints, generated metrics, and fastMRI files are not committed.
- The current five learned ReWave-Net soft-DC weights are approximately
  `[0.772, 0.910, 0.910, 0.938, 0.958]`.

Before publishing a release, follow
[the repository release checklist](docs/repository_release_checklist.md).

## Scope

This repository is research code for accelerated MRI reconstruction. It is not
intended for clinical use.
