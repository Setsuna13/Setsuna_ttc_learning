# Backbone Multiscale Training Runbook

Date: 2026-06-17

## Current Scope

This stage evaluates the new backbone-side multi-level feature fusion only.
The TTC head should stay on the original dot-product/linear-head path for clean attribution.

Keep these head options disabled for backbone-only runs:

```bash
normalize_similarity False similarity_topk_weight 0.0
```

## Smoke / Screening Runs

The current 5% screening runs are:

```text
backbone_ml_full_5p_3e
backbone_ml_no_gates_5p_3e
backbone_ml_context_only_5p_3e
```

All use:

```bash
training_data_ratio 0.05 val_data_ratio 0.05 max_epoch 3 eval_interval 1
```

Purpose:

- `backbone_ml_full_5p_3e`: full multi-level fusion + detail/context/global + gates.
- `backbone_ml_no_gates_5p_3e`: checks whether channel/spatial gates suppress the signal.
- `backbone_ml_context_only_5p_3e`: checks whether pooled context is the main contributor.

## Full Multiscale Backbone Training

After the current screening run finishes, run the full training with multiscale backbone and original head:

```bash
python tools/train.py   -f exp/Deep_TTC.py   -b 8   -d 1   --fp16   -expn full_backbone_ml_original_head   trainset_dir /home/zzqh/TTC/Datasets/train   trainAnnoPath /home/zzqh/TTC/Datasets/train   valset_dir /home/zzqh/TTC/Datasets/val   valAnnoPath /home/zzqh/TTC/Datasets/val   training_data_ratio 1.0   val_data_ratio 1.0   eval_batch_size 4   data_num_workers 0   use_backbone_multiscale_fusion True   use_ms_detail_branch True   use_ms_context_branches True   use_ms_global_branch True   use_ms_channel_gate True   use_ms_spatial_gate True   normalize_similarity False   similarity_topk_weight 0.0
```

Expected output directory:

```text
TTC_outputs/full_backbone_ml_original_head
```

Current detached log:

```text
TTC_outputs/full_backbone_ml_original_head/full_backbone_ml_original_head.nohup.log
```

## Automatic Head Follow-up

The watcher script below waits for `full_backbone_ml_original_head` to finish, checks
that `last_epoch_ckpt.pth` reached epoch 36, and then starts the distribution-head run:

```bash
tools/run_head_after_backbone.sh
```

It writes watcher status to:

```text
TTC_outputs/run_head_after_backbone.watch.log
```

The distribution-head training log will be:

```text
TTC_outputs/full_backbone_ml_distribution_head_50bin/full_backbone_ml_distribution_head_50bin.nohup.log
```

## Next Stage: Head Modification Training

Only start this after `full_backbone_ml_original_head` finishes and the best checkpoint is saved.
Use the same backbone settings so the incremental change is isolated to the head.

Suggested future experiment name:

```text
full_backbone_ml_distribution_head_50bin
```

Command template after the head code is implemented:

```bash
python tools/train.py   -f exp/Deep_TTC.py   -b 8   -d 1   --fp16   -expn full_backbone_ml_distribution_head_50bin   trainset_dir /home/zzqh/TTC/Datasets/train   trainAnnoPath /home/zzqh/TTC/Datasets/train   valset_dir /home/zzqh/TTC/Datasets/val   valAnnoPath /home/zzqh/TTC/Datasets/val   training_data_ratio 1.0   val_data_ratio 1.0   eval_batch_size 4   data_num_workers 0   use_backbone_multiscale_fusion True   use_ms_detail_branch True   use_ms_context_branches True   use_ms_global_branch True   use_ms_channel_gate True   use_ms_spatial_gate True
```

Add the head-specific options to that command once the new head switches are implemented.
