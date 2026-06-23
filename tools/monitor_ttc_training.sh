#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zzqh/TTC/TSTTC}"
EXPECTED_EPOCH="${EXPECTED_EPOCH:-36}"
PYTHON="${PYTHON:-python}"

cd "$ROOT_DIR"

MONITOR_LOG="TTC_outputs/ttc_training_monitor.log"
mkdir -p TTC_outputs

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*" | tee -a "$MONITOR_LOG"
}

ckpt_epoch() {
  local ckpt="$1"
  if [[ ! -f "$ckpt" ]]; then
    echo "-1"
    return
  fi
  "$PYTHON" - "$ckpt" <<'PY'
import sys
try:
    import torch
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    print(int(ckpt.get("start_epoch", -1)))
except Exception:
    print(-1)
PY
}

is_running() {
  local exp="$1"
  pgrep -f "tools/train.py.*-expn ${exp}" >/dev/null 2>&1
}

launch_training() {
  local exp="$1"
  local resume_ckpt="$2"
  local head_mode="$3"
  local exp_dir="TTC_outputs/${exp}"
  local run_log="${exp_dir}/${exp}.monitor_resume.$(date '+%Y%m%d_%H%M%S').log"
  mkdir -p "$exp_dir"

  local args=(
    tools/train.py
    -f exp/Deep_TTC.py
    -b 8
    -d 1
    --fp16
    -expn "$exp"
  )

  if [[ -f "$resume_ckpt" ]]; then
    args+=(--resume -c "$resume_ckpt")
  fi

  args+=(
    trainset_dir /home/zzqh/TTC/Datasets/train
    trainAnnoPath /home/zzqh/TTC/Datasets/train
    valset_dir /home/zzqh/TTC/Datasets/val
    valAnnoPath /home/zzqh/TTC/Datasets/val
    training_data_ratio 1.0
    val_data_ratio 1.0
    eval_batch_size 4
    data_num_workers 0
    use_backbone_multiscale_fusion True
    use_ms_detail_branch True
    use_ms_context_branches True
    use_ms_global_branch True
    use_ms_channel_gate True
    use_ms_spatial_gate True
  )

  if [[ "$head_mode" == "distribution" ]]; then
    args+=(
      head_type distribution
      normalize_similarity False
      similarity_topk_weight 0.0
    )
  elif [[ "$head_mode" == "improved" ]]; then
    args+=(
      head_type bce
      normalize_similarity True
      similarity_topk_ratio 0.05
      similarity_topk_weight 0.4
    )
  else
    args+=(
      head_type bce
      normalize_similarity False
      similarity_topk_weight 0.0
    )
  fi

  log "Launching ${exp}; resume_ckpt=${resume_ckpt}; log=${run_log}"
  setsid -f "$PYTHON" "${args[@]}" > "$run_log" 2>&1
}

BACKBONE_EXP="full_backbone_ml_original_head"
HEAD_EXP="full_backbone_ml_distribution_head_50bin"

BACKBONE_LAST="TTC_outputs/${BACKBONE_EXP}/last_epoch_ckpt.pth"
BACKBONE_LATEST="TTC_outputs/${BACKBONE_EXP}/latest_ckpt.pth"
HEAD_LAST="TTC_outputs/${HEAD_EXP}/last_epoch_ckpt.pth"
HEAD_LATEST="TTC_outputs/${HEAD_EXP}/latest_ckpt.pth"

backbone_epoch="$(ckpt_epoch "$BACKBONE_LAST")"
head_epoch="$(ckpt_epoch "$HEAD_LAST")"

log "Monitor check: ${BACKBONE_EXP} last_epoch=${backbone_epoch}, ${HEAD_EXP} last_epoch=${head_epoch}"

if (( backbone_epoch < EXPECTED_EPOCH )); then
  if is_running "$BACKBONE_EXP"; then
    log "${BACKBONE_EXP} is running; no action."
  else
    log "${BACKBONE_EXP} is not running and is incomplete; resuming."
    launch_training "$BACKBONE_EXP" "$BACKBONE_LATEST" "original"
  fi
  exit 0
fi

log "${BACKBONE_EXP} is complete."

if (( head_epoch >= EXPECTED_EPOCH )); then
  log "${HEAD_EXP} is complete; no action."
  exit 0
fi

if is_running "$HEAD_EXP"; then
  log "${HEAD_EXP} is running; no action."
else
  log "${HEAD_EXP} is not running and is incomplete; resuming/starting."
  launch_training "$HEAD_EXP" "$HEAD_LATEST" "distribution"
fi
