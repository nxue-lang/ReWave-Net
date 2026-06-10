# Scripts

The public repository keeps only the entry points needed to inspect the data,
train and evaluate ReWave-Net, and reproduce its matched Complex U-Net
comparison.

| Script | Purpose |
| --- | --- |
| `_bootstrap.py` | Adds the local `src/` directory to the import path. |
| `inspect_h5_file.py` | Prints the structure of one fastMRI HDF5 file. |
| `test_multifile_dataset.py` | Checks the multi-file dataset output shapes. |
| `test_rewave_net.py` | Tests Haar inversion, band residuals, forward pass, and gradients. |
| `train_rewave_net.py` | Trains ReWave-Net or the matched Complex U-Net baseline. |
| `evaluate_rewave_net.py` | Evaluates a saved ReWave-Net or baseline checkpoint. |
| `summarize_matched_comparison.py` | Reports available matched comparison results. |

## Smoke Test

```bash
python scripts/test_rewave_net.py
```

## Small End-to-End Run

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

## Matched Training

Train ReWave-Net:

```bash
python scripts/train_rewave_net.py \
  --model-type rewave \
  --epochs 20 \
  --num-cascades 5 \
  --base-channels 8 \
  --seed 42 \
  --mask-seed 42
```

Train the Complex U-Net baseline with the same configuration:

```bash
python scripts/train_rewave_net.py \
  --model-type complex \
  --epochs 20 \
  --num-cascades 5 \
  --base-channels 8 \
  --seed 42 \
  --mask-seed 42
```

Evaluate and summarize:

```bash
python scripts/evaluate_rewave_net.py \
  --checkpoint-path outputs/checkpoints/rewave_c5_acc4_best.pt

python scripts/evaluate_rewave_net.py \
  --checkpoint-path outputs/checkpoints/complex_c5_acc4_best.pt

python scripts/summarize_matched_comparison.py
```

Generated files are written under `outputs/`, which is ignored by Git.
