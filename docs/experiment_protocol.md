# Paper-Oriented Experiment Protocol

这份文档用于把当前探索型代码整理成论文式实验流程。核心原则是：最终结论只来自固定 split、固定 mask 规则、固定 metric 定义和可追溯 checkpoint 的结果。

## Primary Claim

推荐把论文主线写成：

```text
Complex-domain reconstruction makes k-space data consistency valid.
Frequency-aware regularization improves low/high-frequency fusion.
KAN-style channel gating further improves adaptive fusion.
Unrolled cascades combine learned regularization with measured k-space consistency.
```

命名上建议使用 `KAN-style gated frequency-aware U-Net`，避免把当前轻量 gate 说成完整 KAN。

## Metric Rule

论文表格里的 PSNR / SSIM / MAE 应使用 target-scale magnitude：

```text
prediction complex image -> magnitude crop
target complex image     -> magnitude crop
both divided by target magnitude max
```

`normalize_to_unit_range` 只用于可视化，不用于论文主表指标。当前正式 complex/unrolled 指标应优先来自：

- `src/mri_recon/evaluation/complex_metrics.py`
- `scripts/evaluate_unrolled_frequency_aware_recon.py`

## Main Ablation

最终 ablation 至少固定以下设置：

- same train/test files
- same acceleration and center fraction
- same mask seed rule
- same number of cascades
- same base channels
- same epochs or same validation/early-stopping policy
- same metric conversion

推荐先跑：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type complex --epochs 20 --num-cascades 5 --base-channels 8 --seed 42 --mask-seed 42
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_complex_recon_c5_acc4_best.pt

python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type fa --epochs 20 --num-cascades 5 --base-channels 8 --seed 42 --mask-seed 42
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_fa_recon_c5_acc4_best.pt

python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type kan --epochs 20 --num-cascades 5 --base-channels 8 --seed 42 --mask-seed 42
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_kan_recon_c5_acc4_best.pt

python scripts\summarize_unrolled_ablation.py
```

`summarize_unrolled_ablation.py` 会输出 config status。只有 `matched` 的模型行适合直接写成严格公平 ablation；`mismatch:*` 只能描述为趋势或 preliminary result。

## Smoke Run

没有 GPU 或只是检查代码时，用：

```powershell
python scripts\train_unrolled_frequency_aware_recon_multifile.py --model-type kan --epochs 1 --num-cascades 2 --base-channels 4 --max-train-files 2 --max-test-files 1 --max-train-samples 8 --max-test-samples 4 --disable-progress
python scripts\evaluate_unrolled_frequency_aware_recon.py --checkpoint-path outputs\checkpoints\unrolled_kan_recon_c2_acc4_best.pt --disable-progress
```

## Figures

主文建议每个方法至少展示：

- target
- zero-filled
- reconstruction
- absolute error map

附录可以再放 soft-DC sweep 和 per-slice improvement histogram。注意 error map 要和 metric 使用同一个 target-scale pair，不要单独 min-max 后再声称对应指标。

## Reporting

主表建议包含：

```text
Method | Cascades | DC | PSNR mean/std | SSIM mean/std | MAE mean/std
```

补充材料建议包含：

- per-slice CSV
- train/test split CSV
- checkpoint args
- config status from ablation summary

当前已有的 early image-domain U-Net 和 naive magnitude hard-DC 结果可以放在方法演进或 discussion 中，用来说明为什么要转向 complex-domain DC，但不建议作为最终主表的核心比较。
