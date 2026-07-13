# Native cross-attention runbook

## Model flow

1. Interleaved reference/current images pass through the TTC backbone at the crop's native resolution.
2. The reference feature map is translated at the same resolution so its box centre matches the current box centre. It is not scaled.
3. Current-frame features form Q; aligned reference-frame features form K and V.
4. Overlapping 3x3 attention searches the reference map at dilation 1, 3, and 6.
5. Expected offsets are decomposed into radial and tangential motion, then combined with appearance statistics to predict the 50-bin scale distribution.
6. The expected scale is converted to TTC and trained with distribution, scale-regression, and TTC-aware losses.

The `dense_qkv` path does not call ROI Align. Native crop resolution is preserved by default; the `dot_product` baseline retains the official 300-pixel downsampling rule for a fair comparison.

## FP-TTC-inspired sparse mode

`fp_sparse_qkv` keeps the same direction and native-resolution crop policy as
`dense_qkv`: the current frame is Q and the centre-aligned reference frame is
K/V. It does not use ROI Align and it does not resize the crop. The difference
is that each current-frame token samples only 2 pyramid levels x 4 reference
points instead of materialising all 3x3 windows at dilation 1, 3, and 6.

The sampling offsets are initialized with deterministic centre/ring anchors and
then learned from the frame pair. A reliability/covisibility gate suppresses
invalid or low-confidence matches before global pooling. The dense mode remains
available, so the two paths can be compared with the same checkpoint and data.

Recommended sparse settings:

```text
cross_attention_mode fp_sparse_qkv
cross_attention_dim 64
cross_attention_heads 2
sparse_attention_levels 2
sparse_attention_points 4
sparse_attention_max_offset 6.0
sparse_attention_offset_scale 2.0
```

For a short screening run rather than the full training, use 5% of the training
set and 10% of validation for three epochs. Keep the seed and all optimizer/loss
settings identical when running the dense control. This screen can reject a bad
design quickly, but a small metric improvement should still be confirmed on a
larger fixed subset.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tools/train.py -f exp/Deep_TTC.py -expn fp_sparse_qkv_quick_sgd_b4_seed0 -b 4 -d 2 -c weights/Deep_TTC.pth --fp16 trainset_dir /home/itslab/zzqh/TSTTC_dataset/train trainAnnoPath /home/itslab/zzqh/TSTTC_dataset/train valset_dir /home/itslab/zzqh/TSTTC_dataset/val valAnnoPath /home/itslab/zzqh/TSTTC_dataset/val training_data_ratio 0.05 val_data_ratio 0.10 eval_batch_size 4 data_num_workers 4 max_epoch 3 warmup_epochs 0 scheduler cos optimizer_name sgd basic_lr_per_img 0.0005 eval_interval 1 print_interval 100 save_history_ckpt False seed 0 freeze_backbone False backbone_lr_scale 0.1 freeze_scale_head True scale_num 50 head_type distribution backbone_type ttcbase use_backbone_multiscale_fusion True cross_attention_mode fp_sparse_qkv cross_attention_dim 64 cross_attention_heads 2 sparse_attention_levels 2 sparse_attention_points 4 sparse_attention_max_offset 6.0 sparse_attention_offset_scale 2.0 cross_attention_dropout 0.0 dense_attention_context_scale 1.0 dense_attention_align_centers True preserve_dense_crop_resolution True cross_attention_reg_loss_weight 0.25 cross_attention_ttc_loss_weight 1.0
```

## Pull and run on the A800 host

```bash
git checkout setsuna/ttc
git pull --ff-only origin setsuna/ttc

TRAIN_DIR=/path/to/train \
VAL_DIR=/path/to/val \
CKPT=/path/to/Deep_TTC_distribution50_best.pth \
BATCH_SIZE=2 \
EVAL_BATCH_SIZE=2 \
bash tools/run_cross_attention_compare.sh
```

The script first evaluates the dot-product checkpoint and then trains dense cross-attention. To skip the baseline evaluation:

```bash
RUN_BASELINE=false bash tools/run_cross_attention_compare.sh
```

The default batch size is conservative because native-resolution crops can exceed 500x500. Increase it only after checking A800 peak memory. Logs are written under `TTC_outputs/dense_qkv_rte_*`.
