#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/run_ablation.sh
#   PYTHON=/home/zzqh/miniconda3/envs/TSTTC/bin/python bash tools/run_ablation.sh
#   EXP_PREFIX=small_ bash tools/run_ablation.sh max_epoch 3 training_data_ratio 0.05 val_data_ratio 0.1
#   BATCH_SIZE=8 DEVICES=1 bash tools/run_ablation.sh max_epoch 12
#   FP16=false bash tools/run_ablation.sh
#   ABLATION_MODE=full bash tools/run_ablation.sh
#   TRAIN_DIR=/path/to/train VAL_DIR=/path/to/val bash tools/run_ablation.sh
#
# Default mode runs the 5 core ablations. ABLATION_MODE=full runs all fine-grained ablations.
# Extra arguments after the script name are appended to every run as Exp opts.

BATCH_SIZE="${BATCH_SIZE:-12}"
PYTHON="${PYTHON:-python}"
DEVICES="${DEVICES:-1}"
EXP_FILE="${EXP_FILE:-exp/Deep_TTC.py}"
ABLATION_MODE="${ABLATION_MODE:-core}"
TRAIN_DIR="${TRAIN_DIR:-/home/zzqh/TTC/Datasets/train}"
VAL_DIR="${VAL_DIR:-/home/zzqh/TTC/Datasets/val}"
FP16="${FP16:-true}"
EXP_PREFIX="${EXP_PREFIX:-}"

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "TRAIN_DIR does not exist: $TRAIN_DIR" >&2
  echo "Set it with: TRAIN_DIR=/path/to/train bash tools/run_ablation.sh" >&2
  exit 1
fi

if [[ ! -d "$VAL_DIR" ]]; then
  echo "VAL_DIR does not exist: $VAL_DIR" >&2
  echo "Set it with: VAL_DIR=/path/to/val bash tools/run_ablation.sh" >&2
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
  valAnnoPath "$VAL_DIR"
)

run_case() {
  local name="$1"
  shift
  echo "============================================================"
  echo "Running ablation: ${EXP_PREFIX}${name}"
  echo "Train: ${TRAIN_DIR}"
  echo "Val: ${VAL_DIR}"
  echo "============================================================"
  "$PYTHON" tools/train.py "${COMMON_ARGS[@]}" -expn "${EXP_PREFIX}${name}" "${DATA_OPTS[@]}" "$@"
}

BASE_OFF=(
  use_backbone_multiscale_fusion False \
  normalize_similarity False \
  similarity_topk_weight 0.0
)

MS_BASE=(
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch False \
  use_ms_context_branches False \
  use_ms_global_branch False \
  use_ms_channel_gate False \
  use_ms_spatial_gate False \
  normalize_similarity False \
  similarity_topk_weight 0.0
)

HEAD_BASE=(
  use_backbone_multiscale_fusion False \
  normalize_similarity False \
  similarity_topk_weight 0.0
)

run_case "abl_00_original_deep" \
  "${BASE_OFF[@]}" \
  "$@"

run_case "abl_01_backbone_ms_only" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  use_ms_spatial_gate True \
  normalize_similarity False \
  similarity_topk_weight 0.0 \
  "$@"

run_case "abl_02_head_norm_topk_only" \
  "${HEAD_BASE[@]}" \
  normalize_similarity True \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4 \
  "$@"

run_case "abl_03_ms_no_gates" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate False \
  use_ms_spatial_gate False \
  normalize_similarity False \
  similarity_topk_weight 0.0 \
  "$@"

run_case "abl_04_full" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  use_ms_spatial_gate True \
  normalize_similarity True \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4 \
  "$@"

if [[ "$ABLATION_MODE" != "full" ]]; then
  exit 0
fi

run_case "abl_01_ms_detail_only" \
  "${MS_BASE[@]}" \
  use_ms_detail_branch True \
  "$@"

run_case "abl_02_ms_context_only" \
  "${MS_BASE[@]}" \
  use_ms_context_branches True \
  "$@"

run_case "abl_03_ms_global_only" \
  "${MS_BASE[@]}" \
  use_ms_global_branch True \
  "$@"

run_case "abl_04_ms_detail_context" \
  "${MS_BASE[@]}" \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  "$@"

run_case "abl_05_ms_no_gates" \
  "${MS_BASE[@]}" \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  "$@"

run_case "abl_06_ms_channel_gate" \
  "${MS_BASE[@]}" \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  "$@"

run_case "abl_07_ms_spatial_gate" \
  "${MS_BASE[@]}" \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_spatial_gate True \
  "$@"

run_case "abl_08_ms_full_no_head" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  use_ms_spatial_gate True \
  normalize_similarity False \
  similarity_topk_weight 0.0 \
  "$@"

run_case "abl_09_head_norm_only" \
  "${HEAD_BASE[@]}" \
  normalize_similarity True \
  "$@"

run_case "abl_10_head_topk_only" \
  "${HEAD_BASE[@]}" \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4 \
  "$@"

run_case "abl_11_head_norm_topk" \
  "${HEAD_BASE[@]}" \
  normalize_similarity True \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4 \
  "$@"

run_case "abl_12_ms_plus_norm" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  use_ms_spatial_gate True \
  normalize_similarity True \
  similarity_topk_weight 0.0 \
  "$@"

run_case "abl_13_ms_plus_topk" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  use_ms_spatial_gate True \
  normalize_similarity False \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4 \
  "$@"

run_case "abl_14_full" \
  use_backbone_multiscale_fusion True \
  use_ms_detail_branch True \
  use_ms_context_branches True \
  use_ms_global_branch True \
  use_ms_channel_gate True \
  use_ms_spatial_gate True \
  normalize_similarity True \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4 \
  "$@"
