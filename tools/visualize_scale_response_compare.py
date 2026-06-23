#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import csv
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from data.ttc_dataset import scale_ratio_to_ttc, ttc_to_scale_ratio
from exp.build import get_exp


def parse_args():
    parser = argparse.ArgumentParser("Visualize scale-bin responses for two TTC models")
    parser.add_argument("-f", "--exp-file", default="exp/Deep_TTC.py")
    parser.add_argument("--baseline-ckpt", default="weights/Deep_TTC.pth")
    parser.add_argument(
        "--multiscale-ckpt",
        default="TTC_outputs/full_backbone_ml_original_head/best_ckpt.pth",
    )
    parser.add_argument("--val-dir", default="/home/zzqh/TTC/Datasets/val")
    parser.add_argument("--output-dir", default="TTC_outputs/scale_response_baseline_vs_multiscale")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--plot-samples", type=int, default=12)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        raise RuntimeError(
            "This PyTorch version does not support torch.load(weights_only=True); "
            "upgrade PyTorch or inspect this checkpoint manually."
        )


def state_dict_from_ckpt(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def make_exp(exp_file, use_multiscale, val_dir, batch_size):
    exp = get_exp(exp_file, None)
    exp.merge([
        "trainset_dir", val_dir,
        "trainAnnoPath", val_dir,
        "valset_dir", val_dir,
        "valAnnoPath", val_dir,
        "eval_batch_size", str(batch_size),
        "data_num_workers", "0",
        "val_data_ratio", "1.0",
        "scale_num", "20",
        "head_type", "bce",
        "use_backbone_multiscale_fusion", str(use_multiscale),
        "normalize_similarity", "False",
        "similarity_topk_weight", "0.0",
    ])
    return exp


def load_model(exp_file, ckpt_path, use_multiscale, val_dir, batch_size, device):
    exp = make_exp(exp_file, use_multiscale, val_dir, batch_size)
    model = exp.get_model().to(device).eval()
    ckpt = safe_torch_load(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state_dict_from_ckpt(ckpt), strict=False)
    return exp, model, list(missing), list(unexpected)


def pred_scale_from_response(response, scale_list, topk):
    k = min(topk, response.shape[-1])
    conf, idx = torch.topk(response, k=k, dim=-1)
    conf = conf / torch.clamp(conf.sum(dim=-1, keepdim=True), min=1e-12)
    return torch.sum(scale_list[idx] * conf, dim=-1)


def anno_id_from_meta(meta):
    if hasattr(meta, "anno_id"):
        return str(meta.anno_id)
    if isinstance(meta, dict):
        return str(meta.get("anno_id", meta.get("id", "")))
    return ""


def collect(args):
    baseline_exp, baseline, base_missing, base_unexpected = load_model(
        args.exp_file, args.baseline_ckpt, False, args.val_dir, args.batch_size, args.device
    )
    ms_exp, multiscale, ms_missing, ms_unexpected = load_model(
        args.exp_file, args.multiscale_ckpt, True, args.val_dir, args.batch_size, args.device
    )
    loader = baseline_exp.get_eval_loader(
        args.batch_size, False, data_path=args.val_dir, anno_path=args.val_dir
    )

    rows = []
    responses = []
    fps = 10 / (baseline_exp.sequence_len - 1)

    with torch.no_grad():
        for imgs, annos, boxes, ttc_gts in loader:
            imgs = imgs.to(args.device)
            boxes = boxes.to(args.device)
            base_out, scale_list, _ = baseline(imgs, boxes, annos)
            ms_out, ms_scale_list, _ = multiscale(imgs, boxes, annos)

            base_resp = base_out.reshape(-1, baseline_exp.scale_num).float().cpu()
            ms_resp = ms_out.reshape(-1, ms_exp.scale_num).float().cpu()
            scale_list = scale_list.float().cpu()
            ms_scale_list = ms_scale_list.float().cpu()

            base_pred_scale = pred_scale_from_response(base_resp, scale_list, args.topk)
            ms_pred_scale = pred_scale_from_response(ms_resp, ms_scale_list, args.topk)
            ttc_gts = torch.as_tensor(ttc_gts).reshape(-1).float().cpu()
            gt_scales = torch.tensor([ttc_to_scale_ratio(float(x), fps=fps) for x in ttc_gts])

            meta = annos.get("metaAnnos", []) if isinstance(annos, dict) else []
            for i in range(base_resp.shape[0]):
                row = {
                    "index": len(rows),
                    "anno_id": anno_id_from_meta(meta[i]) if i < len(meta) else "",
                    "gt_ttc": float(ttc_gts[i]),
                    "gt_scale": float(gt_scales[i]),
                    "baseline_pred_scale": float(base_pred_scale[i]),
                    "multiscale_pred_scale": float(ms_pred_scale[i]),
                    "baseline_pred_ttc": float(scale_ratio_to_ttc(base_pred_scale[i], fps=fps)),
                    "multiscale_pred_ttc": float(scale_ratio_to_ttc(ms_pred_scale[i], fps=fps)),
                    "baseline_peak_scale": float(scale_list[int(torch.argmax(base_resp[i]))]),
                    "multiscale_peak_scale": float(ms_scale_list[int(torch.argmax(ms_resp[i]))]),
                    "baseline_peak_score": float(torch.max(base_resp[i])),
                    "multiscale_peak_score": float(torch.max(ms_resp[i])),
                    "pred_scale_abs_delta": float(abs(base_pred_scale[i] - ms_pred_scale[i])),
                }
                rows.append(row)
                responses.append({
                    "row": row,
                    "baseline": base_resp[i],
                    "multiscale": ms_resp[i],
                    "scale_list": scale_list,
                })
                if len(rows) >= args.num_samples:
                    return rows, responses, {
                        "baseline_missing": base_missing,
                        "baseline_unexpected": base_unexpected,
                        "multiscale_missing": ms_missing,
                        "multiscale_unexpected": ms_unexpected,
                    }

    return rows, responses, {
        "baseline_missing": base_missing,
        "baseline_unexpected": base_unexpected,
        "multiscale_missing": ms_missing,
        "multiscale_unexpected": ms_unexpected,
    }


def plot_samples(responses, path, title, max_samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = responses[:max_samples]
    cols = 3
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.6), dpi=150)
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]

    for ax, item in zip(axes, selected):
        row = item["row"]
        x = item["scale_list"].numpy()
        ax.plot(x, item["baseline"].numpy(), color="#2563eb", lw=1.8, label="baseline")
        ax.plot(x, item["multiscale"].numpy(), color="#dc2626", lw=1.8, label="multiscale")
        ax.axvline(row["gt_scale"], color="#111827", lw=1.2, ls="--", label="gt")
        ax.axvline(row["baseline_pred_scale"], color="#2563eb", lw=1.0, ls=":")
        ax.axvline(row["multiscale_pred_scale"], color="#dc2626", lw=1.0, ls=":")
        ax.set_title(
            "#{idx} gt_ttc={gt:.2f}  delta={delta:.4f}".format(
                idx=row["index"], gt=row["gt_ttc"], delta=row["pred_scale_abs_delta"]
            ),
            fontsize=9,
        )
        ax.set_xlabel("scale ratio")
        ax.set_ylabel("response")
        ax.grid(alpha=0.25)
    for ax in axes[len(selected):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def ttc_bucket(ttc):
    if ttc < 0:
        return "-20~0"
    if ttc < 3:
        return "0~3"
    if ttc < 6:
        return "3~6"
    return "6~20"


def plot_bucket_means(responses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = ["-20~0", "0~3", "3~6", "6~20"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes = axes.reshape(-1)
    for ax, bucket in zip(axes, buckets):
        items = [x for x in responses if ttc_bucket(x["row"]["gt_ttc"]) == bucket]
        if not items:
            ax.axis("off")
            continue
        x = items[0]["scale_list"].numpy()
        base = torch.stack([item["baseline"] for item in items]).mean(0).numpy()
        ms = torch.stack([item["multiscale"] for item in items]).mean(0).numpy()
        gt = sum(item["row"]["gt_scale"] for item in items) / len(items)
        ax.plot(x, base, color="#2563eb", lw=2.0, label="baseline")
        ax.plot(x, ms, color="#dc2626", lw=2.0, label="multiscale")
        ax.axvline(gt, color="#111827", lw=1.2, ls="--", label="mean gt")
        ax.set_title("%s, n=%d" % (bucket, len(items)))
        ax.set_xlabel("scale ratio")
        ax.set_ylabel("mean response")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Mean scale-bin response by GT TTC bucket", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows, responses, meta = collect(args)

    csv_path = os.path.join(args.output_dir, "scale_response_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    first_path = os.path.join(args.output_dir, "scale_response_first_samples.png")
    plot_samples(responses, first_path, "Scale-bin response: first samples", args.plot_samples)

    delta_items = sorted(responses, key=lambda x: x["row"]["pred_scale_abs_delta"], reverse=True)
    delta_path = os.path.join(args.output_dir, "scale_response_largest_delta.png")
    plot_samples(delta_items, delta_path, "Scale-bin response: largest baseline-vs-multiscale deltas", args.plot_samples)

    bucket_path = os.path.join(args.output_dir, "scale_response_bucket_means.png")
    plot_bucket_means(responses, bucket_path)

    meta_path = os.path.join(args.output_dir, "scale_response_meta.json")
    with open(meta_path, "w") as f:
        import json
        json.dump(meta, f, indent=2)

    print("wrote", csv_path)
    print("wrote", first_path)
    print("wrote", delta_path)
    print("wrote", bucket_path)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
