#!/usr/bin/env bash
set -euo pipefail

# Fair comparison: evaluate the official 50-bin ROI/dot-product baseline, then
# train native-resolution target-Q/reference-KV cross-attention without ROI Align.
PYTHON="${PYTHON:-python}"
EXP_FILE="${EXP_FILE:-exp/Deep_TTC.py}"
TRAIN_DIR="${TRAIN_DIR:-/home/zzqh/TTC/Datasets/train}"
VAL_DIR="${VAL_DIR:-/home/zzqh/TTC/Datasets/val}"
CKPT="${CKPT:-weights/Deep_TTC_distribution50_best.pth}"
DEVICES="${DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
FP16="${FP16:-true}"
RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_DENSE="${RUN_DENSE:-${RUN_HYBRID:-true}}"
EXP_PREFIX="${EXP_PREFIX:-dense_qkv_rte_}"
EXTRA_OPTS=("$@")

for path in "$TRAIN_DIR" "$VAL_DIR"; do
  if [[ ! -d "$path" ]]; then
    echo "Dataset directory does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$CKPT" ]]; then
  echo "Warm-start checkpoint does not exist: $CKPT" >&2
  exit 1
fi

FP16_ARGS=()
if [[ "$FP16" == "true" || "$FP16" == "1" || "$FP16" == "yes" ]]; then
  FP16_ARGS+=(--fp16)
fi

MODEL_OPTS=(
  valset_dir "$VAL_DIR"
  valAnnoPath "$VAL_DIR"
  val_data_ratio 1.0
  eval_batch_size "$EVAL_BATCH_SIZE"
  scale_num 50
  head_type distribution
  backbone_type ttcbase
  use_backbone_multiscale_fusion True
  normalize_similarity False
  similarity_topk_weight 0.0
  cross_attention_dim 96
  seed 0
)

if [[ "$RUN_BASELINE" == "true" || "$RUN_BASELINE" == "1" || "$RUN_BASELINE" == "yes" ]]; then
  "$PYTHON" tools/eval.py \
    -f "$EXP_FILE" \
    -expn "${EXP_PREFIX}00_dot_product_eval" \
    -b "$EVAL_BATCH_SIZE" \
    -d "$DEVICES" \
    -c "$CKPT" \
    --allow-partial-load \
    "${FP16_ARGS[@]}" \
    "${MODEL_OPTS[@]}" \
    cross_attention_mode dot_product \
    "${EXTRA_OPTS[@]}"
fi

if [[ "$RUN_DENSE" == "true" || "$RUN_DENSE" == "1" || "$RUN_DENSE" == "yes" ]]; then
  "$PYTHON" tools/train.py \
    -f "$EXP_FILE" \
    -expn "${EXP_PREFIX}01_native_target_q_reference_kv" \
    -b "$BATCH_SIZE" \
    -d "$DEVICES" \
    -c "$CKPT" \
    "${FP16_ARGS[@]}" \
    trainset_dir "$TRAIN_DIR" \
    trainAnnoPath "$TRAIN_DIR" \
    training_data_ratio 1.0 \
    data_num_workers 4 \
    max_epoch 18 \
    warmup_epochs 1 \
    scheduler cos \
    optimizer_name adamw \
    basic_lr_per_img 0.00001 \
    eval_interval 1 \
    print_interval 200 \
    save_history_ckpt False \
    freeze_backbone False \
    backbone_lr_scale 0.1 \
    freeze_scale_head True \
    "${MODEL_OPTS[@]}" \
    cross_attention_mode dense_qkv \
    cross_attention_window_size 3 \
    cross_attention_dilations '(1, 3, 6)' \
    cross_attention_dropout 0.0 \
    dense_attention_context_scale 1.0 \
    dense_attention_align_centers True \
    preserve_dense_crop_resolution True \
    cross_attention_reg_loss_weight 0.25 \
    cross_attention_ttc_loss_weight 1.0 \
    "${EXTRA_OPTS[@]}"
fi

echo "Baseline log: TTC_outputs/${EXP_PREFIX}00_dot_product_eval/val_log.txt"
echo "Dense QKV log: TTC_outputs/${EXP_PREFIX}01_native_target_q_reference_kv/train_log.txt"
