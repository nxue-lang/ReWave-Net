# Scripts Guide

这个目录里保留了不同阶段的训练、评估和检查脚本。不要把它们理解成互相重复的文件，它们记录的是方法逐步演进的过程。

## 推荐阅读顺序

1. `inspect_h5_file.py`: 检查 fastMRI HDF5 文件结构。
2. `run_single_sample_baseline.py`: 单样本 zero-filled reconstruction。
3. `evaluate_zero_filled_baseline.py`: 计算 zero-filled 指标。
4. `train_unet_baseline.py` 和 `evaluate_unet_baseline.py`: image-domain U-Net baseline。
5. `train_complex_unet.py` 和 `evaluate_complex_unet_with_dc.py`: complex-domain U-Net + hard DC。
6. `train_frequency_aware_unet.py` 和 `evaluate_frequency_aware_unet_with_dc.py`: frequency-aware complex U-Net。
7. `train_kan_frequency_aware_unet.py` 和 `evaluate_kan_frequency_aware_unet_soft_dc_sweep.py`: KAN-style gated frequency-aware U-Net。
8. `test_multifile_dataset.py` 和 `train_kan_frequency_aware_unet_multifile.py`: 多文件泛化实验入口。
9. `test_residual_conditioned_wavelet_model.py`: 检查 Haar 可逆性、频段残差和新模型前后向传播。
10. `train_unrolled_frequency_aware_recon_multifile.py` 和 `evaluate_unrolled_frequency_aware_recon.py`: 当前主线，支持 residual-conditioned wavelet unrolled reconstruction。

## Baseline 和检查脚本

| Script | Purpose |
| --- | --- |
| `inspect_h5_file.py` | 打印 HDF5 keys、shape、attributes。 |
| `run_single_sample_baseline.py` | 使用 `configs/single_sample.yaml` 跑一个 zero-filled 样例。 |
| `evaluate_zero_filled_baseline.py` | 对 zero-filled 结果计算 PSNR、SSIM、MAE。 |
| `run_data_consistency_check.py` | 检查 hard data consistency 是否能保持 sampled k-space。 |
| `test_multifile_dataset.py` | 快速检查多文件 dataset 的 sample shape。 |
| `test_residual_conditioned_wavelet_model.py` | 无需数据集的新模型 CUDA/CPU smoke test。 |

## 训练脚本

| Script | Model | Notes |
| --- | --- | --- |
| `train_unet_baseline.py` | `UNet` | 早期 image-domain magnitude baseline。 |
| `train_complex_unet.py` | `ComplexUNet` | complex image 两通道 real/imag baseline。 |
| `train_frequency_aware_recon.py` | `FrequencyAwareComplexRecon` | 不带 U-Net encoder-decoder 的 frequency-aware 尝试。 |
| `train_frequency_aware_unet.py` | `FrequencyAwareComplexUNet` | 当前主要方法之一。 |
| `train_frequency_aware_unet_with_dc.py` | `FrequencyAwareComplexUNet` | 训练时加入 hard DC-aware loss。 |
| `train_kan_frequency_aware_unet.py` | `KANFrequencyAwareComplexUNet` | 单文件 KAN-style gate 尝试。 |
| `train_kan_frequency_aware_unet_multifile.py` | `KANFrequencyAwareComplexUNet` | 多文件历史对照训练入口。 |
| `train_unrolled_frequency_aware_recon_multifile.py` | `UnrolledResidualConditionedWaveletRecon` 及历史 baselines | 多 cascade learned regularizer + soft DC；当前推荐 `--model-type residual_wavelet`。 |

## 评估脚本

| Script | Purpose |
| --- | --- |
| `evaluate_unet_baseline.py` | 比较 zero-filled 和 image-domain U-Net。 |
| `evaluate_unet_with_dc.py` | 保留早期 naive image U-Net + DC 尝试，结果差但有解释价值。 |
| `evaluate_complex_unet_with_dc.py` | 比较 complex U-Net no DC / hard DC。 |
| `evaluate_frequency_aware_recon_with_dc.py` | 比较 frequency-aware recon no DC / hard DC。 |
| `evaluate_frequency_aware_unet_with_dc.py` | 比较 FA-ComplexUNet no DC / hard DC。 |
| `evaluate_frequency_aware_unet_soft_dc_sweep.py` | 扫描 FA-ComplexUNet 的 soft DC strength。 |
| `evaluate_kan_frequency_aware_unet_soft_dc_sweep.py` | 扫描 KAN-style FA-ComplexUNet 的 soft DC strength。 |
| `evaluate_unrolled_frequency_aware_recon.py` | 评估 unrolled 模型，输出 mean/std 和 per-slice metrics。 |
| `summarize_unrolled_evaluation.py` | 汇总 unrolled 评估提升幅度和逐 slice 改善比例。 |
| `summarize_unrolled_ablation.py` | 汇总 unrolled complex / FA / KAN-FA 的 ablation 表。 |
| `summarize_results.py` | 早期 summary 脚本，只汇总部分方法；正式汇总前建议扩展。 |

## 运行建议

单文件 sanity check 可以继续使用默认 `data/knee_singlecoil_val/file1000000.h5`。如果要写正式报告或论文式结果，建议不要只使用这个文件，而是固定 multi-file split 后重新训练和评估。

项目采用 `src/` layout。需要导入本地包的脚本会先调用 `scripts/_bootstrap.py`，所以可以直接运行 `python scripts\xxx.py`；如果要在 notebook、IDE 或其他目录中导入 `mri_recon`，建议先在项目根目录执行：

```powershell
python -m pip install -e .
```

如果 Windows/Anaconda 报 `OMP: Error #15`，先在同一个 PowerShell 里运行：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

常用命令：

```powershell
python scripts\inspect_h5_file.py --h5-path data\knee_singlecoil_val\file1000000.h5
python scripts\evaluate_zero_filled_baseline.py
python scripts\train_complex_unet.py
python scripts\evaluate_complex_unet_with_dc.py
python scripts\train_kan_frequency_aware_unet_multifile.py
python scripts\test_residual_conditioned_wavelet_model.py
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type residual_wavelet --epochs 1 --max-train-files 2 --max-test-files 1 --num-cascades 2 --base-channels 4
```

CPU 上建议先跑：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type residual_wavelet --epochs 3 --num-cascades 2 --base-channels 4 --max-train-files 12 --max-test-files 4 --middle-slice-margin 2 --max-train-samples 60 --max-test-samples 20
```

## 保留的历史尝试

`evaluate_unet_with_dc.py` 这类早期脚本不要急着删。它们说明了一个重要结论：magnitude image-domain 输出不适合直接做 hard k-space DC。把这些放进 ablation 或 discussion，会让你的探索路线更可信。
