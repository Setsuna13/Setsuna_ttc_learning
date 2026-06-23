#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

sys.path.append(os.getcwd())

from exp.build import get_exp


def parse_args():
    parser = argparse.ArgumentParser("Compare baseline and multiscale backbone features")
    parser.add_argument("-f", "--exp-file", default="exp/Deep_TTC.py")
    parser.add_argument("--baseline-ckpt", default="weights/Deep_TTC.pth")
    parser.add_argument(
        "--multiscale-ckpt",
        default="TTC_outputs/full_backbone_ml_original_head/best_ckpt.pth",
    )
    parser.add_argument("--train-dir", default="/home/zzqh/TTC/Datasets/train")
    parser.add_argument("--val-dir", default="/home/zzqh/TTC/Datasets/val")
    parser.add_argument("--output-dir", default="TTC_outputs/feature_compare_baseline_vs_multiscale")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--roi-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--no-heatmaps", action="store_true")
    return parser.parse_args()


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        raise RuntimeError(
            "This PyTorch version does not support torch.load(weights_only=True); "
            "upgrade PyTorch or inspect this checkpoint manually."
        )


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def make_exp(exp_file, use_multiscale):
    exp = get_exp(exp_file, None)
    exp.merge([
        "trainset_dir", "/home/zzqh/TTC/Datasets/train",
        "trainAnnoPath", "/home/zzqh/TTC/Datasets/train",
        "valset_dir", "/home/zzqh/TTC/Datasets/val",
        "valAnnoPath", "/home/zzqh/TTC/Datasets/val",
        "eval_batch_size", "2",
        "data_num_workers", "0",
        "val_data_ratio", "1.0",
        "scale_num", "20",
        "head_type", "bce",
        "use_backbone_multiscale_fusion", str(use_multiscale),
        "normalize_similarity", "False",
        "similarity_topk_weight", "0.0",
    ])
    return exp


def load_model(exp_file, ckpt_path, use_multiscale, device):
    exp = make_exp(exp_file, use_multiscale)
    model = exp.get_model().to(device).eval()
    state = extract_state_dict(safe_torch_load(ckpt_path, device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    return exp, model, list(missing), list(unexpected)


class FeatureCapture:
    def __init__(self, model):
        self.features = {}
        self.handles = []
        modules = {
            "stem": model.backbone.stem,
            "stage2": model.backbone.stage2,
            "stage3_pre_ms": model.backbone.stage3,
        }
        if getattr(model.backbone, "use_multiscale_fusion", False):
            modules["ms_fusion"] = model.backbone.multiscale_fusion
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_module, _inputs, output):
            self.features[name] = output.detach()
        return fn

    def clear(self):
        self.features = {}

    def close(self):
        for handle in self.handles:
            handle.remove()


def summarize_tensor(x):
    x = x.detach().float()
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "abs_mean": x.abs().mean().item(),
        "l2": torch.linalg.vector_norm(x).item(),
        "min": x.min().item(),
        "max": x.max().item(),
    }


def compare_tensors(a, b):
    a = a.detach().float()
    b = b.detach().float()
    if a.shape != b.shape:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    diff = a - b
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    return {
        "cosine": F.cosine_similarity(a_flat, b_flat, dim=1).mean().item(),
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt((diff * diff).mean()).item(),
        "rel_l2": (torch.linalg.vector_norm(diff) / torch.clamp(torch.linalg.vector_norm(a), min=1e-12)).item(),
    }


def target_roi_boxes(head, boxes, feature):
    ref_boxes = boxes[::2, :]
    tar_boxes = boxes[1::2, :]
    _, tar_align_boxes = head.boxes_sample(
        ref_boxes,
        tar_boxes,
        head.get_scale_list(head.scale_number).to(feature.device),
        feature.shape[-2],
        feature.shape[-1],
    )
    return tar_align_boxes


def roi_features(model, feature, boxes, roi_size):
    roi_boxes = target_roi_boxes(model.head, boxes, feature).type_as(feature)
    if roi_boxes.numel() == 0:
        return None
    return roi_align(feature, roi_boxes, (roi_size, roi_size))


def write_heatmap(path, tensor, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = tensor.detach().float().abs().mean(dim=1)[0].cpu()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    im = ax.imshow(arr, cmap="magma")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    heatmap_dir = os.path.join(args.output_dir, "heatmaps")
    if not args.no_heatmaps:
        os.makedirs(heatmap_dir, exist_ok=True)

    baseline_exp, baseline, base_missing, base_unexpected = load_model(
        args.exp_file, args.baseline_ckpt, False, args.device
    )
    ms_exp, multiscale, ms_missing, ms_unexpected = load_model(
        args.exp_file, args.multiscale_ckpt, True, args.device
    )

    baseline_exp.merge([
        "valset_dir", args.val_dir,
        "valAnnoPath", args.val_dir,
        "eval_batch_size", str(args.batch_size),
        "data_num_workers", "0",
    ])
    loader = baseline_exp.get_eval_loader(args.batch_size, False, data_path=args.val_dir, anno_path=args.val_dir)

    base_cap = FeatureCapture(baseline)
    ms_cap = FeatureCapture(multiscale)
    rows = []
    aggregate = defaultdict(list)
    saved_tensors = {}

    with torch.no_grad():
        for batch_idx, (imgs, annos, boxes, _ttc) in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            imgs = imgs.to(args.device)
            boxes = boxes.to(args.device)

            base_cap.clear()
            ms_cap.clear()
            base_final = baseline.backbone(imgs)
            ms_final = multiscale.backbone(imgs)
            base_feats = dict(base_cap.features)
            ms_feats = dict(ms_cap.features)
            base_feats["final"] = base_final.detach()
            ms_feats["final"] = ms_final.detach()

            for layer in sorted(set(base_feats) | set(ms_feats)):
                if layer not in base_feats or layer not in ms_feats:
                    continue
                base_stats = summarize_tensor(base_feats[layer])
                ms_stats = summarize_tensor(ms_feats[layer])
                cmp_stats = compare_tensors(base_feats[layer], ms_feats[layer])
                row = {"batch": batch_idx, "layer": layer, "kind": "global"}
                row.update({"baseline_" + k: v for k, v in base_stats.items()})
                row.update({"multiscale_" + k: v for k, v in ms_stats.items()})
                row.update(cmp_stats)
                rows.append(row)
                for k, v in row.items():
                    if k not in ("batch", "layer", "kind"):
                        aggregate[(layer, "global", k)].append(v)

                base_roi = roi_features(baseline, base_feats[layer], boxes, args.roi_size)
                ms_roi = roi_features(multiscale, ms_feats[layer], boxes, args.roi_size)
                if base_roi is not None and ms_roi is not None:
                    roi_cmp = compare_tensors(base_roi, ms_roi)
                    roi_row = {"batch": batch_idx, "layer": layer, "kind": "target_roi"}
                    roi_row.update({"baseline_" + k: v for k, v in summarize_tensor(base_roi).items()})
                    roi_row.update({"multiscale_" + k: v for k, v in summarize_tensor(ms_roi).items()})
                    roi_row.update(roi_cmp)
                    rows.append(roi_row)
                    for k, v in roi_row.items():
                        if k not in ("batch", "layer", "kind"):
                            aggregate[(layer, "target_roi", k)].append(v)

            if batch_idx == 0:
                if args.save_tensors:
                    saved_tensors = {
                        "baseline": {k: v.detach().cpu() for k, v in base_feats.items()},
                        "multiscale": {k: v.detach().cpu() for k, v in ms_feats.items()},
                    }
                if not args.no_heatmaps:
                    for name in ["stem", "stage2", "stage3_pre_ms", "final"]:
                        if name in base_feats:
                            write_heatmap(os.path.join(heatmap_dir, "baseline_%s.png" % name), base_feats[name], "baseline " + name)
                        if name in ms_feats:
                            write_heatmap(os.path.join(heatmap_dir, "multiscale_%s.png" % name), ms_feats[name], "multiscale " + name)

    base_cap.close()
    ms_cap.close()

    fieldnames = sorted({key for row in rows for key in row.keys()})
    csv_path = os.path.join(args.output_dir, "feature_compare_batches.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for (layer, kind, metric), values in sorted(aggregate.items()):
        summary.append({
            "layer": layer,
            "kind": kind,
            "metric": metric,
            "mean": sum(values) / max(len(values), 1),
            "count": len(values),
        })
    json_path = os.path.join(args.output_dir, "feature_compare_summary.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "baseline_ckpt": args.baseline_ckpt,
                "multiscale_ckpt": args.multiscale_ckpt,
                "baseline_missing": base_missing,
                "baseline_unexpected": base_unexpected,
                "multiscale_missing": ms_missing,
                "multiscale_unexpected": ms_unexpected,
                "summary": summary,
            },
            f,
            indent=2,
        )

    if args.save_tensors:
        torch.save(saved_tensors, os.path.join(args.output_dir, "batch0_features.pt"))

    print("wrote", csv_path)
    print("wrote", json_path)
    if not args.no_heatmaps:
        print("wrote heatmaps under", heatmap_dir)


if __name__ == "__main__":
    main()
