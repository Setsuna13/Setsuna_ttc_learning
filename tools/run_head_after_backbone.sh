#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zzqh/TTC/TSTTC}"
BACKBONE_PID="${BACKBONE_PID:-}"
BACKBONE_EXP="${BACKBONE_EXP:-full_backbone_ml_original_head}"
HEAD_EXP="${HEAD_EXP:-full_backbone_ml_distribution_head_50bin}"
EXPECTED_EPOCH="${EXPECTED_EPOCH:-36}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PYTHON="${PYTHON:-python}"

cd "$ROOT_DIR"

OUT_DIR="TTC_outputs"
BACKBONE_DIR="${OUT_DIR}/${BACKBONE_EXP}"
BACKBONE_CKPT="${BACKBONE_DIR}/last_epoch_ckpt.pth"
HEAD_DIR="${OUT_DIR}/${HEAD_EXP}"
HEAD_LOG="${HEAD_DIR}/${HEAD_EXP}.nohup.log"
WATCH_LOG="${OUT_DIR}/run_head_after_backbone.watch.log"

mkdir -p "$HEAD_DIR"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*" | tee -a "$WATCH_LOG"
}

if [[ -z "$BACKBONE_PID" ]]; then
  BACKBONE_PID="$(pgrep -f "tools/train.py.*-expn ${BACKBONE_EXP}" | head -n 1 || true)"
fi

if [[ -z "$BACKBONE_PID" ]]; then
  log "No running process found for ${BACKBONE_EXP}; checking checkpoint directly."
else
  log "Waiting for ${BACKBONE_EXP} process ${BACKBONE_PID} to finish."
  while kill -0 "$BACKBONE_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  log "${BACKBONE_EXP} process ${BACKBONE_PID} finished."
fi

if [[ ! -f "$BACKBONE_CKPT" ]]; then
  log "Missing checkpoint: ${BACKBONE_CKPT}. Head training will not start."
  exit 1
fi

FINISHED_EPOCH="$("$PYTHON" -c 'import sys, torch; ckpt=torch.load(sys.argv[1], map_location="cpu"); print(int(ckpt.get("start_epoch", -1)))' "$BACKBONE_CKPT")"

if (( FINISHED_EPOCH < EXPECTED_EPOCH )); then
  log "Backbone checkpoint epoch ${FINISHED_EPOCH} < expected ${EXPECTED_EPOCH}. Head training will not start."
  exit 1
fi

log "Backbone training reached epoch ${FINISHED_EPOCH}; starting ${HEAD_EXP}."
log "Head log: ${HEAD_LOG}"

"$PYTHON" tools/train.py \
  -f exp/Deep_TTC.py \
  -b 8 \
  -d 1 \
  --fp16 \
  -expn "$HEAD_EXP" \
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
  normalize_similarity True \
  similarity_topk_ratio 0.05 \
  similarity_topk_weight 0.4 \
  > "$HEAD_LOG" 2>&1

log "${HEAD_EXP} finished."
