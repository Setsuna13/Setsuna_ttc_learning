# Native dense cross-attention runbook

## Model flow

1. Interleaved reference/current images pass through the TTC backbone at the crop's native resolution.
2. The reference feature map is translated at the same resolution so its box centre matches the current box centre. It is not scaled.
3. Current-frame features form Q; aligned reference-frame features form K and V.
4. Overlapping 3x3 attention searches the reference map at dilation 1, 3, and 6.
5. Expected offsets are decomposed into radial and tangential motion, then combined with appearance statistics to predict the 50-bin scale distribution.
6. The expected scale is converted to TTC and trained with distribution, scale-regression, and TTC-aware losses.

The `dense_qkv` path does not call ROI Align. Native crop resolution is preserved by default; the `dot_product` baseline retains the official 300-pixel downsampling rule for a fair comparison.

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
