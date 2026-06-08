# MRI Reconstruction Pipeline

这个项目是一个 fastMRI single-coil knee MRI 重建实验管线。当前重点不是把所有尝试都“收成一个最终答案”，而是保留你从 baseline 到 complex-domain、data consistency、frequency-aware block、KAN-style gate 的探索路径，让别人能看懂你为什么一步步这么做。

## 当前主线

推荐把现在的方法理解成下面这条线：

```text
raw k-space
  -> Cartesian undersampling mask
  -> zero-filled complex image
  -> complex reconstruction network / unrolled reconstruction cascades
  -> k-space data consistency
  -> cropped magnitude image
  -> PSNR / SSIM / MAE
```

这里面最值得继续发展的版本是：

1. `ComplexUNet`: 先把重建从 image-domain magnitude 转到 complex image space。
2. `FrequencyAwareComplexUNet`: 加入低频结构分支、高频细节分支和自适应 gate。
3. `KANFrequencyAwareComplexUNet`: 保留 frequency-aware 思路，用轻量 KAN-style channel gate 做融合。
4. `MultiFileFastMRIComplexSingleCoilDataset`: 从单个 volume 的 slice split 走向多文件泛化实验。
5. `UnrolledFrequencyAwareRecon` / `UnrolledKANFrequencyAwareRecon`: 把 learned regularizer 和 k-space DC 交替堆叠成多 cascade reconstruction，这是下一轮主线。

## 项目结构

```text
configs/               小样本实验配置
data/                  fastMRI knee single-coil 数据
docs/                  方法说明和实验脉络
outputs/               已生成的 checkpoint、figure、metric、split
scripts/               训练、评估、检查脚本
src/mri_recon/         可复用的数据、模型、指标、变换和 DC 代码
```

建议先看：

1. `docs/method_notes.md`: 方法演进、当前结论、后续优先级。
2. `scripts/README.md`: 每个脚本的用途和推荐运行顺序。
3. `src/mri_recon/models/`: 模型实现。
4. `src/mri_recon/data/`: 数据集和 mask 逻辑。

## 环境准备

如果是第一次在新环境运行，建议先安装依赖和本地包：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

为了方便直接运行 `scripts/*.py`，需要导入本地包的脚本会通过 `scripts/_bootstrap.py` 把 `src` 加到 Python import path。正式环境里更推荐使用 `pip install -e .`。

如果在 Windows/Anaconda 环境遇到 `OMP: Error #15`，可以在当前 PowerShell 临时设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

这是本地运行环境的 OpenMP runtime 冲突问题，不是 MRI 模型代码本身的问题。

## 快速检查

```powershell
python scripts\inspect_h5_file.py --h5-path data\knee_singlecoil_val\file1000000.h5
python scripts\run_single_sample_baseline.py
python scripts\evaluate_zero_filled_baseline.py
python scripts\evaluate_unet_baseline.py
```

如果要继续做更像“正式实验”的方向，优先跑多文件版本：

```powershell
python scripts\test_multifile_dataset.py
python scripts\train_kan_frequency_aware_unet_multifile.py
```

如果要跑和主流 MoDL/VarNet 更接近的新版本，从 unrolled 入口开始：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type kan --num-cascades 5 --base-channels 8
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_kan_frequency_aware_recon_c5_acc4_best.pt
```

如果没有 GPU，先不要直接跑上面的全量命令。建议用这个 CPU 友好版本确认趋势：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type kan --epochs 3 --num-cascades 2 --base-channels 4 --max-train-files 12 --max-test-files 4 --middle-slice-margin 2 --max-train-samples 60 --max-test-samples 20
```

## 当前结果快照

这些数字来自当前 `outputs/metrics`，主要是 `file1000000.h5` 的验证 slice。它们适合用来说明方法探索趋势，但还不适合作为最终泛化结论。

| Method | PSNR | SSIM | MAE | Notes |
| --- | ---: | ---: | ---: | --- |
| zero-filled | 21.33 | 0.555 | 0.0722 | 单文件 4x undersampling 对照 |
| image U-Net | 25.15 | 0.514 | 0.0439 | PSNR/MAE 提升，SSIM 没赢 zero-filled |
| complex U-Net + hard DC | 23.82 | 0.539 | 0.0523 | complex-domain 后 DC 才更合理 |
| FA-ComplexUNet + hard DC | 22.06 | 0.551 | 0.0660 | SSIM 接近 zero-filled，PSNR 不强 |
| KAN-style FA + soft DC, lambda=0.10 | 25.71 | 0.540 | 0.0407 | 当前单文件 PSNR/MAE 最好 |
| Unrolled KAN-style FA, 5 cascades | 26.44 | 0.575 | 0.0391 | 多文件 test split，440 slices，target-scale metrics |

最新 unrolled 结果来自 `outputs/metrics/unrolled_frequency_aware_recon_metrics.csv`：

```text
zero-filled:      PSNR 25.3741 +/- 2.2399, SSIM 0.5455 +/- 0.0920, MAE 0.042351 +/- 0.011801
unrolled KAN-FA:  PSNR 26.4432 +/- 2.8353, SSIM 0.5752 +/- 0.1041, MAE 0.039139 +/- 0.013267
```

在 `440` 个 test slices 中，unrolled 方法相对 zero-filled 的改进覆盖：

```text
PSNR improved: 412 / 440 slices
SSIM improved: 439 / 440 slices
MAE improved:  400 / 440 slices
```

## 需要谨慎解释的点

- 目前很多评估脚本会对每张预测图和 target 单独做 min-max normalization，这会削弱强度尺度错误对指标的影响。
- 一部分结果来自单个 volume 内部的 slice split，泛化结论需要以多文件 split 为准。
- `KANFrequencyAwareComplexUNet` 是 KAN-style gate，不是完整 Kolmogorov-Arnold Network。报告里建议这样命名，避免被问到完整 KAN 的 spline/basis 实现细节。
- 早期的 image-domain U-Net + hard DC 指标很差，这不是失败文件，反而是有价值的 ablation：它说明 DC 应该在 complex/k-space 一致的表示里做。

## 下一步建议

优先级最高的是把“公平比较”补齐：同一 train/test split、同一 mask 规则、同一 normalization 规则，比较 zero-filled、complex U-Net、FA-ComplexUNet、KAN-style FA-ComplexUNet，再分别看 no DC / soft DC / hard DC。

第二步再补更强的指标和图：每个方法给 mean/std，至少展示 3 个 test volume 的 representative slices，并放 error map。这样你的方法就会从“我试了很多东西”变成“我有一条清楚的实验逻辑”。

新的 unrolled 脚本已经开始朝这个方向走：评估会输出 mean/std 和 per-slice metrics，并且 metric conversion 使用 target scale，而不是对 prediction 单独 min-max。

## Ablation Plan

为了回答“提升到底来自哪里”，下一步固定同一个 multi-file split 做 unrolled ablation：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type complex --epochs 20 --num-cascades 5 --base-channels 8
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_complex_recon_c5_acc4_best.pt

python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type fa --epochs 20 --num-cascades 5 --base-channels 8
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_fa_recon_c5_acc4_best.pt

python scripts\summarize_unrolled_ablation.py
```

已有的 KAN-style unrolled 结果会被自动纳入 `summarize_unrolled_ablation.py`。这组 ablation 能分别回答：

```text
Unrolled Complex U-Net vs zero-filled: unrolled + DC 是否有效
Unrolled FA vs Unrolled Complex U-Net: frequency-aware 是否有效
Unrolled KAN-FA vs Unrolled FA: KAN-style gate 是否有效
```

当前快速 ablation 结果：

| Method | Setting | PSNR | SSIM | MAE | Delta vs zero-filled |
| --- | --- | ---: | ---: | ---: | --- |
| zero-filled | same test split | 25.3741 | 0.5455 | 0.042351 | baseline |
| Unrolled Complex U-Net | c3, 10 epochs | 25.1662 | 0.5412 | 0.043516 | -0.2079 dB |
| Unrolled FA Complex U-Net | c3, 10 epochs | 25.8299 | 0.5563 | 0.040883 | +0.4558 dB |
| Unrolled KAN-FA Complex U-Net | c5, 20 epochs | 26.4432 | 0.5752 | 0.039139 | +1.0691 dB |

注意：这张表目前是快速趋势表，不是完全公平的最终 ablation，因为 KAN-FA 使用了更深 cascade 和更长训练。严格结论需要把 complex / FA / KAN-FA 都统一到相同 `num_cascades`、epoch、train/test split 和 validation policy。
