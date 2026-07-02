#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zzqh/TTC/TSTTC}"
EXP="${EXP:-full_backbone_ml_distribution_head_50bin}"
EXPECTED_EPOCH="${EXPECTED_EPOCH:-36}"
PYTHON="${PYTHON:-python}"
POLL_SECONDS="${POLL_SECONDS:-43200}"
MAX_RESTARTS="${MAX_RESTARTS:-0}" # 0 means unlimited restarts.

cd "$ROOT_DIR"

EXP_DIR="TTC_outputs/${EXP}"
WATCH_LOG="${WATCH_LOG:-${EXP_DIR}/watch_current_training.log}"
TRAIN_LOG="${TRAIN_LOG:-${EXP_DIR}/train_log.txt}"
RUN_LOG="${RUN_LOG:-${EXP_DIR}/auto_resume.nohup.log}"
LOCK_DIR="${EXP_DIR}/watch_current_training.lock"

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

is_complete() {
  local last_epoch
  last_epoch="$(ckpt_epoch "${EXP_DIR}/last_epoch_ckpt.pth")"
  [[ "$last_epoch" =~ ^-?[0-9]+$ ]] && (( last_epoch >= EXPECTED_EPOCH ))
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

launch_resume() {
  local resume_ckpt="$1"

  if [[ -z "$resume_ckpt" || ! -f "$resume_ckpt" ]]; then
    log "No checkpoint found for ${EXP}; not starting from scratch."
    return 1
  fi

  log "Launching resume from ${resume_ckpt}; run log=${RUN_LOG}"
  setsid -f "$PYTHON" tools/train.py \
    -f exp/Deep_TTC.py \
    -b 8 \
    -d 1 \
    --fp16 \
    -expn "$EXP" \
    --resume \
    -c "$resume_ckpt" \
    trainset_dir /home/zzqh/TTC/Datasets/train \
    trainAnnoPath /home/zzqh/TTC/Datasets/train \
    valset_dir /home/zzqh/TTC/Datasets/val \
    valAnnoPath /home/zzqh/TTC/Datasets/val \
    training_data_ratio 1.0 \
    val_data_ratio 1.0 \
    eval_batch_size 4 \
    data_num_workers 0 \
    use_backbone_multiscale_fusion True \
    use_ms_detail_branch True \
    use_ms_context_branches True \
    use_ms_global_branch True \
    use_ms_channel_gate True \
    use_ms_spatial_gate True \
    head_type distribution \
    normalize_similarity False \
    similarity_topk_weight 0.0 \
    >> "$RUN_LOG" 2>&1
}

restarts=0
log "Watcher started for ${EXP}; expected_epoch=${EXPECTED_EPOCH}; poll=${POLL_SECONDS}s; max_restarts=${MAX_RESTARTS}"

while true; do
  if is_complete; then
    log "Experiment is complete; watcher exiting."
    exit 0
  fi

  if is_training_running; then
    sleep "$POLL_SECONDS"
    continue
  fi

  if (( MAX_RESTARTS > 0 && restarts >= MAX_RESTARTS )); then
    log "Training is stopped and incomplete, but max restarts reached (${MAX_RESTARTS}); watcher exiting."
    exit 1
  fi

  resume_ckpt="$(select_resume_ckpt)"
  resume_epoch="$(ckpt_epoch "$resume_ckpt")"
  log "Training stopped and incomplete; resume attempt $((restarts + 1)) from epoch=${resume_epoch}."
  launch_resume "$resume_ckpt"
  restarts=$((restarts + 1))
  sleep "$POLL_SECONDS"
done
