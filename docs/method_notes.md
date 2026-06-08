# Method Notes

这份文件记录方法演进和下一步建议，目的是保留你的尝试过程，同时让项目主线更清晰。

## 1. 问题设定

当前数据是 fastMRI knee single-coil validation set。基本任务是：

```text
fully-sampled k-space
  -> simulated Cartesian undersampling
  -> zero-filled reconstruction
  -> learned reconstruction
  -> compare with reconstruction_esc
```

当前 mask 设置主要是 acceleration `4`，center fraction `0.08`。

## 2. 尝试路线

### Step A: Zero-filled baseline

相关文件：

- `scripts/evaluate_zero_filled_baseline.py`
- `src/mri_recon/baselines.py`
- `src/mri_recon/sampling.py`

作用：建立最基本的下限。这个 baseline 很重要，因为 MRI 重建不是普通去噪任务，zero-filled 的 aliasing pattern 是所有后续模型要修正的对象。

### Step B: Image-domain U-Net

相关文件：

- `src/mri_recon/models/unet.py`
- `scripts/train_unet_baseline.py`
- `scripts/evaluate_unet_baseline.py`

结果观察：PSNR 和 MAE 提升明显，但 SSIM 没有超过 zero-filled。说明 image-domain U-Net 能减小平均误差，但可能过度平滑或改变结构纹理。

### Step C: Image-domain U-Net + hard DC

相关文件：

- `scripts/evaluate_unet_with_dc.py`

结果观察：`unet_dc` PSNR 降到大约 `13.89`。这条尝试应该保留，因为它说明把 magnitude U-Net 输出硬塞回 k-space 做 DC 不合理。DC 应该在 complex image 或 k-space 表示中完成。

### Step D: Complex U-Net + hard DC

相关文件：

- `src/mri_recon/models/complex_unet.py`
- `src/mri_recon/data/complex_dataset.py`
- `scripts/train_complex_unet.py`
- `scripts/evaluate_complex_unet_with_dc.py`

意义：这是方法转折点。模型输出 real/imag 两个通道，DC 可以在 Fourier domain 替换 sampled k-space。这个方向比 image-domain DC 更符合 MRI 物理约束。

### Step E: Frequency-aware reconstruction

相关文件：

- `src/mri_recon/models/frequency_aware_recon.py`
- `src/mri_recon/models/frequency_aware_unet.py`
- `scripts/train_frequency_aware_recon.py`
- `scripts/train_frequency_aware_unet.py`

核心想法：

- low-frequency branch: 负责平滑结构和全局 anatomy。
- high-frequency branch: 负责边缘、纹理和细节。
- gate: 自适应融合两者。

这个想法可以作为你自己的方法主线来讲，因为它是针对 MRI k-space/频率特性的 inductive bias。

### Step F: Train-time DC 和 soft DC sweep

相关文件：

- `scripts/train_frequency_aware_unet_with_dc.py`
- `scripts/evaluate_frequency_aware_unet_soft_dc_sweep.py`
- `scripts/evaluate_kan_frequency_aware_unet_soft_dc_sweep.py`

结果观察：soft DC 的强度不是越大越好。当前 KAN-style FA 版本在 `lambda=0.10` 附近 PSNR/MAE 最好，而 `lambda=1.0` 的 hard DC 更偏向提高 SSIM，但会损失 PSNR。

### Step G: KAN-style frequency gate

相关文件：

- `src/mri_recon/models/kan_frequency_aware_unet.py`
- `scripts/train_kan_frequency_aware_unet.py`
- `scripts/train_kan_frequency_aware_unet_multifile.py`

建议命名：`KAN-style gated frequency-aware U-Net`。当前实现是轻量非线性 gate，不是完整 KAN。这样写更准确，也更容易防住答辩时的追问。

### Step H: Unrolled frequency-aware reconstruction

相关文件：

- `src/mri_recon/reconstruction/torch_ops.py`
- `src/mri_recon/models/unrolled_frequency_aware.py`
- `scripts/train_unrolled_frequency_aware_recon_multifile.py`
- `scripts/evaluate_unrolled_frequency_aware_recon.py`

这是下一阶段主线。它把原来的 FA/KAN-style U-Net 从“一次性重建网络”升级成 model-based unrolled reconstruction：

```text
zero-filled complex image
  -> FA/KAN learned regularizer
  -> soft k-space data consistency
  -> FA/KAN learned regularizer
  -> soft k-space data consistency
  -> ...
```

这个结构和 MoDL / VarNet 的思想更接近：网络不只是直接 hallucinate 一个图，而是在每个 cascade 后回到 measured k-space 做物理一致性约束。

当前实现里每个 cascade 有一个 learnable soft DC weight。初始化默认是 `0.1`，训练时会学习每一层应该多强地相信 measured k-space。

## 3. 当前最重要的问题

### Normalization

当前很多地方使用 `normalize_to_unit_range` 对单张图单独归一化。这样适合可视化，但用于 PSNR/SSIM 时会掩盖强度尺度误差。

建议后续把归一化分成两类：

- visualization normalization: 只用于保存图。
- metric normalization: 使用 target 的固定 scale，或 fastMRI metadata 中的 `max` / `norm`，保持 prediction 和 target 在同一尺度下比较。

### 数据切分

单文件内 slice split 可以做 sanity check，但不能证明泛化。后续推荐以 `outputs/splits/multifile_split_seed42.csv` 或重新生成的固定 multi-file split 为主。

### 公平比较

正式比较时，每个模型应该使用：

- 同一批 test files。
- 同一 acceleration 和 center fraction。
- 同一 mask seed 规则。
- 同一 metric normalization。
- 同一 checkpoint 选择规则。

## 4. 建议的下一轮实验

1. 先固定 multi-file split，评估 zero-filled baseline。
2. 在同一 split 上训练并评估 complex U-Net。
3. 在同一 split 上训练并评估 FrequencyAwareComplexUNet。
4. 在同一 split 上训练并评估 KAN-style FA-ComplexUNet。
5. 训练并评估 unrolled FA/KAN-style model。
6. 对每个模型做 no DC / soft DC / hard DC 或 internal DC 对照。
7. 输出 mean/std，而不只是 mean。
8. 每个方法展示至少 3 个 volume 的重建图和 error map。

推荐先跑一个小规模 smoke run：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --epochs 1 --max-train-files 2 --max-test-files 1 --num-cascades 2 --base-channels 4
```

确认没问题后，再跑正式版本：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type kan --num-cascades 5 --base-channels 8
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_kan_frequency_aware_recon_c5_acc4_best.pt
```

如果当前机器没有 CUDA/GPU，先跑 CPU 友好版本：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type kan --epochs 3 --num-cascades 2 --base-channels 4 --max-train-files 12 --max-test-files 4 --middle-slice-margin 2 --max-train-samples 60 --max-test-samples 20
```

## 5. 可以在汇报里讲的故事

一个清楚的叙事方式是：

```text
Zero-filled baseline gives structured aliasing artifacts.
Image U-Net reduces pixel error but does not respect k-space physics.
Naive hard DC on magnitude images fails, showing the need for complex-domain consistency.
Complex U-Net enables valid k-space DC.
Frequency-aware blocks inject MRI-specific low/high frequency bias.
KAN-style gating improves adaptive fusion, with soft DC balancing fidelity and perceptual structure.
Unrolled cascades make the method closer to model-based MRI reconstruction by alternating learned regularization and measured k-space consistency.
```

这条线能把你的“试错过程”讲成方法设计过程，而不是杂乱尝试。
