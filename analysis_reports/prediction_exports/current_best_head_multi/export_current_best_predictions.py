#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from core.auxil import configure_module, time_synchronized  # noqa: E402
from data.ttc_dataset import scale_ratio_to_ttc, ttc_to_scale_ratio  # noqa: E402
from exp.build import get_exp  # noqa: E402


DEFAULT_OPTS = [
    "trainset_dir", "/home/zzqh/TTC/Datasets/train",
    "trainAnnoPath", "/home/zzqh/TTC/Datasets/train",
    "valset_dir", "/home/zzqh/TTC/Datasets/val",
    "valAnnoPath", "/home/zzqh/TTC/Datasets/val",
    "training_data_ratio", "1.0",
    "val_data_ratio", "1.0",
    "eval_batch_size", "4",
    "data_num_workers", "0",
    "use_backbone_multiscale_fusion", "True",
    "use_ms_detail_branch", "True",
    "use_ms_context_branches", "True",
    "use_ms_global_branch", "True",
    "use_ms_channel_gate", "True",
    "use_ms_spatial_gate", "True",
    "head_type", "distribution",
    "normalize_similarity", "False",
    "similarity_topk_weight", "0.0",
]


def ttc_bin(value):
    if -20 < value < 0:
        return "ttc -20~0"
    if 0 <= value < 3:
        return "ttc 0~3"
    if 3 <= value < 6:
        return "ttc 3~6"
    if 6 <= value < 20:
        return "ttc 6~20"
    return "out_of_range"


def relative_ttc_error(pred_ttc, gt_ttc):
    pred_tmp = max(min(float(pred_ttc), 20.0), -20.0)
    gt_tmp = max(min(float(gt_ttc), 20.0), -20.0)
    div = abs(gt_tmp) if abs(gt_tmp) > 1e-4 else 1e-4
    return abs((pred_tmp - gt_tmp) / div) * 100.0


def get_batch_fps(annos, reference, sequence_len):
    frame_gap = annos.get("frame_gap") if isinstance(annos, dict) else None
    if frame_gap is None:
        frame_gap = torch.full(
            (reference.numel(),),
            sequence_len - 1,
            dtype=reference.dtype,
            device=reference.device,
        )
    else:
        frame_gap = torch.as_tensor(frame_gap, dtype=reference.dtype, device=reference.device).view(-1)
        if frame_gap.numel() == 0:
            frame_gap = torch.full_like(reference.view(-1), sequence_len - 1)
        elif frame_gap.numel() == 1 and reference.numel() != 1:
            frame_gap = frame_gap.expand(reference.numel())
        elif frame_gap.numel() != reference.numel():
            raise ValueError(
                "frame_gap count {} does not match prediction count {}".format(
                    frame_gap.numel(), reference.numel()
                )
            )
    frame_gap = torch.clamp(frame_gap, min=1)
    return 10.0 / frame_gap, frame_gap


def anno_field(anno, name, default=""):
    if hasattr(anno, name):
        return getattr(anno, name)
    if isinstance(anno, dict):
        return anno.get(name, default)
    try:
        return anno[name]
    except Exception:
        return default


def parse_args():
    parser = argparse.ArgumentParser("Export current best TSTTC predictions and ground truth")
    parser.add_argument(
        "--ckpt",
        default=str(REPO_ROOT / "TTC_outputs/head+multi/best_ckpt.pth"),
        help="checkpoint path",
    )
    parser.add_argument("--exp-file", default="exp/Deep_TTC.py")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "analysis_reports/prediction_exports/current_best_head_multi"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_module()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "current_best_predictions.tsv"
    summary_path = out_dir / "current_best_predictions_summary.txt"
    meta_path = out_dir / "current_best_predictions_meta.json"

    exp = get_exp(args.exp_file, None)
    exp.box_level = True
    exp.merge(DEFAULT_OPTS + (args.opts or []))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = exp.get_model().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.fp16:
        model = model.half()

    evaluator = exp.get_evaluator(args.batch_size, is_distributed=False)
    dataloader = evaluator.dataloader
    scale_number = exp.scale_num
    rows = []
    per_bin = defaultdict(lambda: {"n": 0, "rel_sum": 0.0, "abs_sum": 0.0})
    inference_time = 0.0
    timed_batches = max(len(dataloader) - 1, 1)
    tensor_dtype = torch.float16 if args.fp16 else torch.float32

    start = time.time()
    with torch.no_grad():
        for cur_iter, batch in enumerate(dataloader):
            imgs, annos, candidate_boxes, ttc_gts = batch
            if imgs is None:
                continue
            imgs = imgs.to(device=device, dtype=tensor_dtype, non_blocking=True)
            is_time_record = cur_iter < len(dataloader) - 1
            if is_time_record:
                infer_start = time.time()
            outputs, scale_list, pred_scales = model.forward(imgs, candidate_boxes, annos)
            if is_time_record:
                inference_time += time_synchronized() - infer_start

            outputs = outputs.reshape(-1, scale_number)
            scale_list = torch.as_tensor(scale_list, dtype=outputs.dtype, device=outputs.device)
            if pred_scales is None:
                row_sums = outputs.sum(dim=-1, keepdim=True)
                is_distribution = torch.allclose(
                    row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3
                )
                if is_distribution:
                    pred_scales = torch.sum(outputs * scale_list, dim=-1)
                else:
                    pred_conf, pred_bin = torch.topk(outputs, k=args.topk, dim=-1)
                    pred_conf = pred_conf / torch.clamp(pred_conf.sum(dim=-1, keepdim=True), min=1e-12)
                    pred_scales = torch.sum(scale_list[pred_bin] * pred_conf, dim=-1)
            else:
                pred_scales = pred_scales.reshape(-1).type_as(outputs)

            fps, frame_gap = get_batch_fps(annos, pred_scales, exp.sequence_len)
            pred_ttcs = scale_ratio_to_ttc(pred_scales, fps)
            gt_ttcs = ttc_gts.to(device=pred_ttcs.device, dtype=pred_ttcs.dtype).view(-1)
            gt_scales = ttc_to_scale_ratio(gt_ttcs, fps)

            pred_ttcs_cpu = pred_ttcs.detach().float().cpu()
            gt_ttcs_cpu = gt_ttcs.detach().float().cpu()
            pred_scales_cpu = pred_scales.detach().float().cpu()
            gt_scales_cpu = gt_scales.detach().float().cpu()
            frame_gap_cpu = frame_gap.detach().float().cpu()
            fps_cpu = fps.detach().float().cpu()

            meta_annos = annos.get("metaAnnos", []) if isinstance(annos, dict) else []
            for i in range(pred_ttcs_cpu.numel()):
                pred_ttc = float(pred_ttcs_cpu[i])
                gt_ttc = float(gt_ttcs_cpu[i])
                abs_err = abs(pred_ttc - gt_ttc)
                rel_err = relative_ttc_error(pred_ttc, gt_ttc)
                bin_name = ttc_bin(gt_ttc)
                meta = meta_annos[i] if i < len(meta_annos) else None
                rec = {
                    "index": len(rows),
                    "anno_id": anno_field(meta, "anno_id", ""),
                    "ttc_bin": bin_name,
                    "frame_gap": int(round(float(frame_gap_cpu[i]))),
                    "fps": float(fps_cpu[i]),
                    "gt_ttc": gt_ttc,
                    "pred_ttc": pred_ttc,
                    "abs_ttc_error": abs_err,
                    "rel_ttc_error_pct": rel_err,
                    "gt_scale": float(gt_scales_cpu[i]),
                    "pred_scale": float(pred_scales_cpu[i]),
                    "abs_scale_error": abs(float(pred_scales_cpu[i]) - float(gt_scales_cpu[i])),
                    "img_path": anno_field(meta, "img_path", ""),
                    "seq_id": anno_field(meta, "seq_id", ""),
                    "obj_id": anno_field(meta, "obj_id", ""),
                }
                rows.append(rec)
                per_bin[bin_name]["n"] += 1
                per_bin[bin_name]["rel_sum"] += rel_err
                per_bin[bin_name]["abs_sum"] += abs_err

    fieldnames = [
        "index",
        "anno_id",
        "ttc_bin",
        "frame_gap",
        "fps",
        "gt_ttc",
        "pred_ttc",
        "abs_ttc_error",
        "rel_ttc_error_pct",
        "gt_scale",
        "pred_scale",
        "abs_scale_error",
        "img_path",
        "seq_id",
        "obj_id",
    ]
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total_rel = sum(row["rel_ttc_error_pct"] for row in rows) / max(len(rows), 1)
    total_abs = sum(row["abs_ttc_error"] for row in rows) / max(len(rows), 1)
    lines = [
        "Current best prediction export",
        "================================",
        f"checkpoint: {args.ckpt}",
        f"checkpoint_best_rte: {ckpt.get('best_rte')}",
        f"checkpoint_start_epoch: {ckpt.get('start_epoch')}",
        f"num_samples: {len(rows)}",
        f"average_relative_ttc_error_pct: {total_rel:.6f}",
        f"average_abs_ttc_error: {total_abs:.6f}",
        f"average_inference_time_per_batch: {inference_time / timed_batches:.6f}",
        f"elapsed_seconds: {time.time() - start:.2f}",
        "",
        "Per-bin summary",
        "---------------",
    ]
    for key in ["ttc -20~0", "ttc 0~3", "ttc 3~6", "ttc 6~20", "out_of_range"]:
        item = per_bin.get(key, {"n": 0, "rel_sum": 0.0, "abs_sum": 0.0})
        n = item["n"]
        if n:
            lines.append(
                f"{key}: n={n}, avg_rel_ttc_error_pct={item['rel_sum'] / n:.6f}, "
                f"avg_abs_ttc_error={item['abs_sum'] / n:.6f}"
            )
        else:
            lines.append(f"{key}: n=0")
    lines.extend([
        "",
        "Columns",
        "-------",
        "gt_ttc: validation ground-truth TTC",
        "pred_ttc: model predicted TTC from predicted scale",
        "rel_ttc_error_pct: abs(clamp(pred)-clamp(gt))/abs(clamp(gt))*100",
        "gt_scale/pred_scale: scale ratios using per-sample frame_gap fps",
    ])
    summary_path.write_text("\n".join(lines) + "\n")

    meta = {
        "checkpoint": args.ckpt,
        "checkpoint_best_rte": ckpt.get("best_rte"),
        "checkpoint_start_epoch": ckpt.get("start_epoch"),
        "exp_file": args.exp_file,
        "opts": DEFAULT_OPTS + (args.opts or []),
        "output_tsv": str(tsv_path),
        "output_summary": str(summary_path),
        "num_samples": len(rows),
        "average_relative_ttc_error_pct": total_rel,
        "average_abs_ttc_error": total_abs,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(tsv_path)
    print(summary_path)
    print(meta_path)


if __name__ == "__main__":
    main()
