#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zzqh/TTC/TSTTC}"
EXP="${EXP:-per_bin_residual_calib_frozen_full_v2}"
EXPECTED_EPOCH="${EXPECTED_EPOCH:-6}"
POLL_SECONDS="${POLL_SECONDS:-43200}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
PYTHON="${PYTHON:-python}"
BASE_CKPT="${BASE_CKPT:-/home/zzqh/TTC/TSTTC/TTC_outputs/head+multi/best_ckpt.pth}"

cd "$ROOT_DIR"

EXP_DIR="TTC_outputs/${EXP}"
WATCH_LOG="${WATCH_LOG:-${EXP_DIR}/watch_12h.log}"
RUN_LOG="${RUN_LOG:-${EXP_DIR}/watch_12h_resume.nohup.log}"
LOCK_DIR="${EXP_DIR}/watch_12h.lock"

mkdir -p "$EXP_DIR"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*" | tee -a "$WATCH_LOG"
}

if mkdir "$LOCK_DIR" 2>/dev/null; then
  trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
else
  log "Watcher lock exists at ${LOCK_DIR}; another watcher is probably running. Exiting."
  exit 0
fi

ckpt_epoch() {
  local ckpt="$1"
  if [[ ! -f "$ckpt" ]]; then
    echo "-1"
    return
  fi
  "$PYTHON" - "$ckpt" <<'PYCKPT'
import sys
try:
    import torch
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    print(int(ckpt.get("start_epoch", -1)))
except Exception:
    print(-1)
PYCKPT
}

is_training_running() {
  pgrep -f "tools/train.py.*-expn ${EXP}" >/dev/null 2>&1
}

select_resume_ckpt() {
  local candidates=(
    "${EXP_DIR}/last_epoch_ckpt.pth"
    "${EXP_DIR}/latest_ckpt.pth"
  )
  local best_ckpt=""
  local best_epoch=-1
  local ckpt epoch

  for ckpt in "${candidates[@]}" "${EXP_DIR}"/epoch_*_ckpt.pth; do
    [[ -f "$ckpt" ]] || continue
    epoch="$(ckpt_epoch "$ckpt")"
    [[ "$epoch" =~ ^-?[0-9]+$ ]] || epoch=-1
    if (( epoch > best_epoch )); then
      best_epoch="$epoch"
      best_ckpt="$ckpt"
    fi
  done

  printf '%s\n' "$best_ckpt"
}

latest_epoch() {
  local ckpt
  ckpt="$(select_resume_ckpt)"
  ckpt_epoch "$ckpt"
}

is_complete() {
  local epoch
  epoch="$(latest_epoch)"
  [[ "$epoch" =~ ^-?[0-9]+$ ]] && (( epoch >= EXPECTED_EPOCH ))
}

launch_training() {
  local ckpt="$1"
  shift
  local resume_args=()

  if [[ -f "$ckpt" ]]; then
    resume_args=(--resume -c "$ckpt")
    log "Launching resume from ${ckpt}; run log=${RUN_LOG}"
  else
    if [[ ! -f "$BASE_CKPT" ]]; then
      log "No resume checkpoint and BASE_CKPT missing: ${BASE_CKPT}; not launching."
      return 1
    fi
    resume_args=(-c "$BASE_CKPT")
    log "No resume checkpoint found; launching fine-tune from base ${BASE_CKPT}; run log=${RUN_LOG}"
  fi

  setsid -f "$PYTHON" tools/train.py \
    -f exp/Deep_TTC.py \
    -expn "$EXP" \
    -b 8 \
    -d 1 \
    --fp16 \
    "${resume_args[@]}" \
    trainset_dir /home/zzqh/TTC/Datasets/train \
    trainAnnoPath /home/zzqh/TTC/Datasets/train \
    valset_dir /home/zzqh/TTC/Datasets/val \
    valAnnoPath /home/zzqh/TTC/Datasets/val \
    training_data_ratio 1.0 \
    val_data_ratio 1.0 \
    eval_batch_size 4 \
    data_num_workers 0 \
    max_epoch "$EXPECTED_EPOCH" \
    warmup_epochs 0 \
    eval_interval 1 \
    save_history_ckpt False \
    print_interval 200 \
    scheduler cos \
    use_backbone_multiscale_fusion True \
    use_ms_detail_branch True \
    use_ms_context_branches True \
    use_ms_global_branch True \
    use_ms_channel_gate True \
    use_ms_spatial_gate True \
    head_type distribution \
    normalize_similarity False \
    similarity_topk_weight 0.0 \
    use_per_bin_residual_head True \
    residual_bin_num 31 \
    residual_scale_range 0.03 \
    residual_loss_weight 0.3 \
    final_scale_loss_weight 0.2 \
    residual_short_loss_weight 0.02 \
    residual_mid_ttc_abs_thresh 3.0 \
    residual_mid_loss_weight 0.25 \
    residual_long_ttc_abs_thresh 6.0 \
    residual_long_loss_weight 1.0 \
    residual_tail_ttc_abs_thresh 12.0 \
    residual_tail_loss_weight 1.2 \
    freeze_backbone True \
    freeze_scale_head True \
    basic_lr_per_img 0.000025 \
    >> "$RUN_LOG" 2>&1
}

restarts=0
log "12h watcher started for ${EXP}; expected_epoch=${EXPECTED_EPOCH}; poll=${POLL_SECONDS}s; max_restarts=${MAX_RESTARTS}"

while true; do
  current_epoch="$(latest_epoch)"

  if is_complete; then
    log "Experiment complete at checkpoint epoch=${current_epoch}; watcher exiting."
    exit 0
  fi

  if is_training_running; then
    log "Training is running; latest checkpoint epoch=${current_epoch}; next check in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  if (( MAX_RESTARTS > 0 && restarts >= MAX_RESTARTS )); then
    log "Training stopped before completion, but max restarts reached (${MAX_RESTARTS}); watcher exiting."
    exit 1
  fi

  resume_ckpt="$(select_resume_ckpt)"
  resume_epoch="$(ckpt_epoch "$resume_ckpt")"
  log "Training is not running and incomplete; restart attempt $((restarts + 1)) from epoch=${resume_epoch}."
  launch_training "$resume_ckpt"
  restarts=$((restarts + 1))
  sleep "$POLL_SECONDS"
done
