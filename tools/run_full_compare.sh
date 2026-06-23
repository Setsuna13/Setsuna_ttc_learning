#!/usr/bin/env bash
set -euo pipefail

# Full-data comparison:
#   1. original deep baseline
#   2. full multi-scale + similarity aggregation improvement
#
# Extra arguments after the script name are appended to both runs as Exp opts.

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
PYTHON="${PYTHON:-python}"
DEVICES="${DEVICES:-1}"
EXP_FILE="${EXP_FILE:-exp/Deep_TTC.py}"
TRAIN_DIR="${TRAIN_DIR:-/home/zzqh/TTC/Datasets/train}"
VAL_DIR="${VAL_DIR:-/home/zzqh/TTC/Datasets/val}"
FP16="${FP16:-true}"
EXP_PREFIX="${EXP_PREFIX:-fullcmp_}"
EXTRA_OPTS=("$@")

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "TRAIN_DIR does not exist: $TRAIN_DIR" >&2
  exit 1
fi

if [[ ! -d "$VAL_DIR" ]]; then
  echo "VAL_DIR does not exist: $VAL_DIR" >&2
  exit 1
fi

COMMON_ARGS=(-f "$EXP_FILE" -b "$BATCH_SIZE" -d "$DEVICES")
if [[ "$FP16" == "true" || "$FP16" == "1" || "$FP16" == "yes" ]]; then
  COMMON_ARGS+=(--fp16)
fi

DATA_OPTS=(
  trainset_dir "$TRAIN_DIR" \
  trainAnnoPath "$TRAIN_DIR" \
  valset_dir "$VAL_DIR" \
  valAnnoPath "$VAL_DIR" \
  training_data_ratio 1.0 \
  val_data_ratio 1.0 \
  eval_batch_size "$EVAL_BATCH_SIZE" \
  data_num_workers 0
)

run_case() {
  local name="$1"
  shift
  echo "============================================================"
  echo "Running full comparison: ${EXP_PREFIX}${name}"
  echo "Batch size: ${BATCH_SIZE}, eval batch size: ${EVAL_BATCH_SIZE}"
  echo "Train: ${TRAIN_DIR}"
  echo "Val: ${VAL_DIR}"
  echo "============================================================"
  "$PYTHON" tools/train.py "${COMMON_ARGS[@]}" -expn "${EXP_PREFIX}${name}" "${DATA_OPTS[@]}" "$@" "${EXTRA_OPTS[@]}"
}

run_case "00_original_deep" \
  use_backbone_multiscale_fusion False \
  normalize_similarity False \
  similarity_topk_weight 0.0

run_case "01_full_multiscale" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  use_ms_spatial_gate True \
  normalize_similarity True \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4
