#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zzqh/TTC/TSTTC}"
EXP="${EXP:-full_backbone_ml_distribution_head_50bin}"
EXPECTED_EPOCH="${EXPECTED_EPOCH:-36}"
PYTHON="${PYTHON:-python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"

cd "$ROOT_DIR"

EXP_DIR="TTC_outputs/${EXP}"
WATCH_LOG="${EXP_DIR}/epoch2_resume_watch.log"
RUN_LOG="${EXP_DIR}/epoch2_auto_resume.nohup.log"
LATEST_CKPT="${EXP_DIR}/latest_ckpt.pth"
LAST_CKPT="${EXP_DIR}/last_epoch_ckpt.pth"
mkdir -p "$EXP_DIR"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*" | tee -a "$WATCH_LOG"
}

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

best_epoch() {
  local last_epoch latest_epoch
  last_epoch="$(ckpt_epoch "$LAST_CKPT")"
  latest_epoch="$(ckpt_epoch "$LATEST_CKPT")"
  if (( last_epoch > latest_epoch )); then
    echo "$last_epoch"
  else
    echo "$latest_epoch"
  fi
}

is_training_running() {
  pgrep -f "python tools/train.py.*-expn ${EXP}" >/dev/null 2>&1
}

launch_resume() {
  if [[ ! -f "$LATEST_CKPT" ]]; then
    log "No latest checkpoint at ${LATEST_CKPT}; cannot resume."
    return 1
  fi

  log "Launching resume from ${LATEST_CKPT}; run log=${RUN_LOG}"
  setsid -f "$PYTHON" tools/train.py \
    -f exp/Deep_TTC.py \
    -expn "$EXP" \
    -b 8 \
    -d 1 \
    --fp16 \
    --resume \
    -c "$LATEST_CKPT" \
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
log "Watcher started for ${EXP}; expected_epoch=${EXPECTED_EPOCH}; poll=${POLL_SECONDS}s"

while true; do
  epoch="$(best_epoch)"
  if (( epoch >= EXPECTED_EPOCH )); then
    log "Experiment complete at checkpoint epoch=${epoch}; watcher exiting."
    exit 0
  fi

  if is_training_running; then
    sleep "$POLL_SECONDS"
    continue
  fi

  if (( restarts >= MAX_RESTARTS )); then
    log "Training is stopped and incomplete at checkpoint epoch=${epoch}, but max restarts reached (${MAX_RESTARTS}); watcher exiting."
    exit 1
  fi

  log "Training is stopped and incomplete at checkpoint epoch=${epoch}; resume attempt $((restarts + 1))/${MAX_RESTARTS}."
  launch_resume
  restarts=$((restarts + 1))
  sleep "$POLL_SECONDS"
done
