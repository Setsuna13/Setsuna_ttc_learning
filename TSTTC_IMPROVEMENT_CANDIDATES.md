# TSTTC Improvement Candidates

本文档记录当前 TSTTC 实验的已知现象、已验证方案、失败原因和后续可尝试的改进方向。目标是尽量提升验证集 RTE，同时避免破坏当前已经较好的 `0~3s` 区间。

## 1. 当前基线

当前最优模型：

```text
checkpoint: /home/zzqh/TTC/TSTTC/TTC_outputs/head+multi/best_ckpt.pth
Average RTE: 10.420%
```

分段表现大致为：

```text
ttc -20~0: 12.01%
ttc 0~3:    5.51%
ttc 3~6:    7.13%
ttc 6~20:  11.63%
```

旧 PPM 最优：

```text
path: /home/zzqh/TTC/zzqh_old_version/TSTTC_bak/TTC_outputs/Date_all/PPM1st/best_ckpt.pth
best_rte: 10.393%
```

当前多尺度版本相比旧 PPM：

- `0~3s` 有提升趋势。
- 总体 RTE 没明显超过旧 PPM。
- 主要被 `6~20s`、`-20~0s` 和少量长尾大误差样本拉回去。

## 2. 当前预测数据结论

预测导出文件：

```text
/home/zzqh/TTC/TSTTC/analysis_reports/prediction_exports/current_best_head_multi/current_best_predictions.tsv
```

关键统计：

```text
num_samples: 29087
mean RTE: 10.420%
median RTE: 6.19%
p90 RTE: 21.21%
p95 RTE: 28.06%
```

误差贡献：

```text
ttc 6~20: 样本占 42.0%，误差贡献 46.9%
ttc -20~0: 样本占 29.4%，误差贡献 33.9%
ttc 0~3: 样本占 13.3%，误差贡献 7.0%
```

长尾影响很大：

```text
RTE >= 20%: 11.43% samples, 44.32% total error
RTE >= 100%: 0.59% samples, about 12.9% total error
```

结论：

- 当前不是 `0~3s` 没学好。
- 优先方向应该是 `6~20s`、`-20~0s` 和长尾异常样本。
- 直接继续优化近距离区间，容易伤到已经较好的结果。

## 3. 已验证但不推荐的方案

### 3.1 Eval-time robust box crop

已加入默认关闭开关：

```text
use_robust_box_crop = False
```

测试命令使用当前 best checkpoint，只在 eval 时把疑似异常 bbox 替换成邻近帧平滑 bbox。

结果：

```text
baseline current best RTE: 10.420%
robust box crop RTE:       11.293%
```

分段退化：

```text
ttc -20~0: 12.01 -> 12.34
ttc 0~3:    5.51 -> 8.33
ttc 3~6:    7.13 -> 8.69
ttc 6~20:  11.63 -> 12.08
```

详细报告：

```text
/home/zzqh/TTC/TSTTC/analysis_reports/prediction_exports/current_best_head_multi/robust_box_crop_ablation_summary.txt
```

结论：

- 这个方法能救少数极端坏框样本，例如：

```text
idx 27832: 789.8% -> 9.6%
idx 15934: 862.1% -> 36.4%
idx 3930:  703.1% -> 42.7%
```

- 但它会把更多原本预测好的样本改坏。
- 原因是当前模型是在原始 crop 分布下训练的，eval 时直接换 crop 会造成输入分布偏移。
- 不建议直接打开 `use_robust_box_crop=True` 做最终评估。

## 4. 下一步优先推荐方案

### 4.1 Prediction fallback / gated robust crop

优先级：高

核心思想：不要对所有疑似 bbox 样本替换 crop，而是只在原始预测明显发散时触发二次预测。

建议规则：

```text
先用原始 crop 预测。
如果满足以下任一条件，再使用 robust crop 重新预测：
1. |pred_ttc_raw| > 20
2. abs(pred_scale - 1.0) < 0.005
3. pred_ttc 与简单 scale/box trend 方向强冲突
```

最终输出可选：

```text
if trigger_fallback:
    pred = robust_pred
else:
    pred = original_pred
```

也可以更保守：

```text
if trigger_fallback and abs(robust_pred) <= 20:
    pred = robust_pred
else:
    pred = original_pred
```

为什么推荐：

- `idx 27832`、`15934`、`3930` 这类极端 outlier 明显能被 robust crop 救回来。
- 但大部分普通样本不能动。
- gating 可以只处理发散预测，避免伤 `0~3s` 大盘。

风险：

- 如果 gating 太宽，会复现 robust crop 的整体退化。
- 如果 gating 太窄，收益有限。

验证方式：

1. 基于当前 TSV 找出 `|pred_ttc| > 20` 或 `pred_scale` 接近 1 的样本。
2. 只对这些样本运行 robust crop 二次预测。
3. 合并原始预测和 fallback 预测，离线计算 RTE。
4. 如果离线 RTE < 10.420%，再考虑集成到 evaluator。

### 4.2 Per-sample old PPM vs current multi-scale gate

优先级：高

需要先导出旧 PPM `10.393` 模型的逐样本预测 TSV，格式与当前模型一致。

目标：

```text
current multi-scale 对哪些样本更好？
old PPM 对哪些样本更好？
两者是否互补？
```

建议步骤：

1. 用旧 PPM best checkpoint 导出预测 TSV。
2. 按 `anno_id` 合并当前 TSV 和旧 PPM TSV。
3. 分析：

```text
current better count
PPM better count
both bad count
current-only rescued bins
PPM-only rescued bins
```

4. 训练或手写一个轻量 gate，输入只使用推理时可获得的信息：

```text
pred_ttc_current
pred_scale_current
distribution entropy
top1/top2 confidence gap
bbox area ratio
bbox center shift
occ_ratio if available
```

禁止使用 GT bin 做推理 gate，因为那是泄漏。

为什么推荐：

- 当前多尺度对 `0~3s` 有价值。
- 旧 PPM 总体仍略优。
- 如果两者错误不重合，ensemble/gating 可能比继续改 backbone 更快突破 10.39。

### 4.3 Hard-tail fine-tune

优先级：中高

当前大误差长尾很少，但贡献很高。

候选策略：

```text
从当前 best 或旧 PPM best 继续 fine-tune 3~5 epoch。
低学习率。
对以下样本轻微加权：
- ttc 6~20
- ttc -20~0
- gt_ttc > 16
- gt_ttc < -17
- 历史预测 RTE >= 20% 的样本
```

建议权重不要太大：

```text
normal: 1.0
6~20: 1.2~1.4
-20~0: 1.1~1.3
hard-tail: 1.3~1.6
0~3: 1.0
```

风险：

- 可能伤 `0~3s`。
- 如果 hard-tail 中有脏标注，过拟合会更严重。

建议先做：

```text
只对 clean hard-tail 加权。
把 bbox 明显异常或 occ_ratio 高的 hard-tail 降权，而不是加权。
```

### 4.4 Long-TTC calibration / residual head

优先级：中

当前长 TTC 存在 range compression：

```text
6~20s: pred_ttc 平均偏小，尤其 16~20s
-20~0s: 极端负 TTC 被压向 0
```

建议增加一个轻量 residual/calibration head：

```text
pred_scale_base -> pred_ttc_base
residual = small_mlp(features or similarity stats)
pred_ttc_final = pred_ttc_base + residual
```

训练时只对长区间加弱约束：

```text
if |gt_ttc| > 6:
    use residual loss
else:
    residual loss weight small or zero
```

风险：

- 如果 residual 作用到全区间，容易伤 `0~3s`。
- 需要严格做 ablation。


已落地的开关式实现：

```text
use_per_bin_residual_head=False  # 默认关闭，旧 checkpoint 严格兼容
residual_bin_num=31
residual_scale_range=0.03
residual_loss_weight=0.3
final_scale_loss_weight=0.5
```

当前实现不是全局残差，而是每个 coarse scale/bin 都预测一个 residual 分布：

```text
P(i, k) = P_base(scale_bin=i) * P_residual(residual_bin=k | scale_bin=i)
final_scale = sum_i sum_k P(i, k) * clamp(scale_bin_i + residual_k)
```

建议训练入口：

```bash
python tools/train.py -f exp/Deep_TTC.py -expn per_bin_residual_calib_frozen_full_v2 \
  -b 8 -d 1 --fp16 -c /home/zzqh/TTC/TSTTC/TTC_outputs/head+multi/best_ckpt.pth \
  trainset_dir /home/zzqh/TTC/Datasets/train trainAnnoPath /home/zzqh/TTC/Datasets/train \
  valset_dir /home/zzqh/TTC/Datasets/val valAnnoPath /home/zzqh/TTC/Datasets/val \
  training_data_ratio 1.0 val_data_ratio 1.0 eval_batch_size 4 data_num_workers 0 \
  max_epoch 6 warmup_epochs 0 eval_interval 1 save_history_ckpt False print_interval 200 \
  scheduler cos basic_lr_per_img 0.000025 \
  use_per_bin_residual_head True residual_bin_num 31 residual_scale_range 0.03 \
  residual_loss_weight 0.3 final_scale_loss_weight 0.2 \
  residual_short_loss_weight 0.02 residual_mid_ttc_abs_thresh 3.0 residual_mid_loss_weight 0.25 \
  residual_long_ttc_abs_thresh 6.0 residual_long_loss_weight 1.0 \
  residual_tail_ttc_abs_thresh 12.0 residual_tail_loss_weight 1.2 \
  freeze_backbone True freeze_scale_head True
```

注意：不要用 `--resume` 从旧 checkpoint 跑这个实验，因为 optimizer 和新增 head 参数不匹配。用 `-c` 做 fine-tune 初始化即可。当前实现会零初始化新 residual head，并可冻结 backbone 与原 `scale_preds`，把实验限定为校准层学习。短 TTC 的 residual loss 权重很低，主要让长 TTC / tail 学 residual。

实践记录：`per_bin_residual_calib_frozen_full` 首次启动时因为 `max_epoch=6` 小于默认 `no_aug_epochs=8`，YOLOX warm-cos scheduler 全程落在最低 LR，已停止。正式运行的是 `per_bin_residual_calib_frozen_full_v2`，使用 `scheduler cos`。

### 4.5 Non-uniform scale bins / TTC-aware bins

优先级：中

当前 50-bin head 直接加 bin 数已经验证收益不明显。问题可能不是 bin 数，而是 bin 分布对长 TTC 不友好。

候选方向：

```text
scale bin 不均匀分布。
在 scale 接近 1 的区域加密，因为长 TTC 对应 scale 变化非常小。
或者按 TTC 空间均匀采样，再映射到 scale。
```

为什么可能有效：

- `6~20s` 和 `out_of_range` 对 scale 的微小误差非常敏感。
- 均匀 scale bin 可能不适合长 TTC。

风险：

- 会改变 head label/输出解释，需要重新训练。
- 可能影响旧指标复现。

建议只作为第二阶段实验。

## 5. 数据质量建议

### 5.1 保留异常样本清单，不直接删除

已经生成异常复查图和 TSV：

```text
/home/zzqh/TTC/TSTTC/analysis_reports/prediction_exports/current_best_head_multi/anomaly_image_review/
```

包括：

```text
top RTE ref/tar pairs
low RTE ref/tar pairs
bbox-enriched prediction TSV
```

建议不要直接删除异常样本，因为验证集指标需要可比性。

更稳的做法：

```text
训练时降权。
分析时单独报告 clean / suspicious 两套指标。
评估主指标仍保留原始 val。
```

### 5.2 bbox anomaly flag

可以增加只读 flag，不改变 crop：

```text
box_occ_high
box_area_jump
box_height_jump
box_center_jump
```

用途：

- 训练 sample weight。
- 错误分析分组。
- fallback gate 特征。

不要一开始就用这些 flag 替换 crop。

## 6. 推荐实验顺序

### Experiment A: Fallback robust crop offline merge

预期成本：低

步骤：

1. 用当前 best 原始 TSV 作为 base。
2. 只对触发条件样本重新预测 robust crop。
3. 合并预测结果。
4. 离线计算 RTE。

推荐触发条件第一版：

```text
abs(pred_ttc_raw) > 20
or abs(pred_scale - 1.0) < 0.005
```

成功标准：

```text
Average RTE < 10.420%
0~3s 不明显退化
```

### Experiment B: Old PPM vs current multi-scale complementarity

预期成本：中

步骤：

1. 导出旧 PPM TSV。
2. 合并 current TSV。
3. 看 oracle best-of-two 上限。
4. 如果 oracle 上限明显好，再做 gate。

成功标准：

```text
oracle best-of-two 明显低于 10.39
```

### Experiment C: Conservative hard-tail fine-tune

预期成本：中高

步骤：

1. 固定当前 best 或 old PPM best。
2. hard-tail 加权。
3. suspicious bbox 降权。
4. 只训 3~5 epoch。

成功标准：

```text
6~20s 和 -20~0s 改善
0~3s 不退化超过 0.2~0.3 RTE
总 RTE < 10.39
```

## 7. 当前不建议优先做的事

不建议：

```text
1. 直接全量开启 robust box crop eval。
2. 继续盲目增加 bin 数。
3. 继续堆更复杂 backbone，但没有 per-sample 错误对比。
4. 删除 val 异常样本后报告主指标。
5. 用 GT bin 做推理 gate。
```

原因：

- 这些做法要么已经验证无收益，要么会破坏指标可比性。

## 8. 当前最推荐下一步

最推荐做：

```text
Experiment A: fallback robust crop offline merge
```

因为它最小、最快，而且直接利用了已经观察到的现象：

- robust crop 能救一部分极端 outlier。
- 但不能影响普通样本。
- 只在原始预测发散时触发，可能保留当前 best 的大盘表现，同时修掉少量长尾灾难样本。

如果 Experiment A 没有收益，再做 Experiment B：旧 PPM 与当前多尺度逐样本互补分析。
