#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

from exp.build import get_exp
from core.auxil import configure_module
from data.ttc_dataset import scale_ratio_to_ttc, ttc_to_scale_ratio


def make_parser():
    parser = argparse.ArgumentParser("Dump cross-attention matrices for TTC head")
    parser.add_argument("-f", "--exp_file", default="exp/Deep_TTC.py", type=str)
    parser.add_argument("-c", "--ckpt", required=True, type=str)
    parser.add_argument("--val-dir", required=True, type=str)
    parser.add_argument("--out-dir", default="outputs/cross_attention_dump", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-samples", default=16, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify Exp options, same style as train/eval.py",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _common_tokens(head, xin, tar_boxes, ref_boxes):
    _, H, W = xin.shape[-3:]
    attn_grid = max(1, min(int(head.cross_attention_grid_size), head.grid_size))
    context_scale = max(float(head.max_scale), 1.0)
    if head.shift:
        context_scale *= (head.grid_size + head.shift_kernel_size - 1) / head.grid_size

    ref_maps = xin[::2, ]
    tar_maps = xin[1::2, ]
    ref_tokens = head.sample_box_tokens(ref_maps, ref_boxes, attn_grid, H, W, context_scale)
    tar_tokens = head.sample_box_tokens(tar_maps, tar_boxes, attn_grid, H, W, 1.0)

    position_embed = head.position_proj(
        head.grid_position_tokens(attn_grid, xin.device, xin.dtype)
    )
    ref_tokens = head.ref_proj(ref_tokens) + head.token_type_embed.weight[0].view(1, 1, -1) + position_embed
    tar_tokens = head.tar_proj(tar_tokens) + head.token_type_embed.weight[1].view(1, 1, -1) + position_embed
    pair_geometry = head.box_pair_geometry(ref_boxes, tar_boxes, H, W).type_as(xin)
    geometry_embed = head.geometry_proj(pair_geometry)
    return ref_tokens, tar_tokens, pair_geometry, geometry_embed, attn_grid


@torch.no_grad()
def ref_to_target_debug(head, xin, tar_boxes, ref_boxes):
    ref_tokens, tar_tokens, pair_geometry, geometry_embed, attn_grid = _common_tokens(
        head, xin, tar_boxes, ref_boxes
    )
    ref_tokens = head.token_norm(ref_tokens + geometry_embed.unsqueeze(1))
    tar_tokens = head.token_norm(tar_tokens + geometry_embed.unsqueeze(1))

    q = head.q_proj(head.query_norm(ref_tokens))
    k = head.k_proj(tar_tokens)
    v = head.v_proj(tar_tokens)

    q = head.split_attention_heads(q)
    k = head.split_attention_heads(k)
    v = head.split_attention_heads(v)

    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(head.cross_attention_head_dim, 1))
    attn_logits = attn_logits * head.attn_logit_scale.exp().clamp(max=10.0)
    attn_heads = F.softmax(attn_logits, dim=-1)
    context = torch.matmul(attn_heads, v)
    context = head.merge_attention_heads(context)
    match_tokens = head.context_norm(ref_tokens + head.out_proj(context))

    match_mean = match_tokens.mean(dim=1)
    match_max = match_tokens.max(dim=1).values
    ref_mean = ref_tokens.mean(dim=1)
    tar_mean = tar_tokens.mean(dim=1)
    delta_mean = (match_tokens - ref_tokens).abs().mean(dim=1)
    pair_feature = torch.cat([match_mean, match_max, ref_mean, tar_mean, delta_mean, geometry_embed], dim=-1)
    raw_scale = head.scale_regression_head(pair_feature).squeeze(-1)
    pred_scale = torch.sigmoid(raw_scale) * (head.max_scale - head.min_scale) + head.min_scale
    attn = attn_heads.mean(dim=1)
    attn_entropy = -(attn * (attn + 1e-12).log()).sum(dim=-1).mean(dim=-1)
    return {
        "mode": "ref_to_target",
        "pred_scale": pred_scale,
        "raw_scale": raw_scale,
        "attn": attn,
        "attn_heads": attn_heads,
        "attn_entropy": attn_entropy,
        "attn_grid": attn_grid,
        "pair_geometry": pair_geometry,
        "scale_list": head.get_scale_list(head.scale_number).to(device=xin.device, dtype=xin.dtype),
    }


@torch.no_grad()
def scale_query_debug(head, xin, tar_boxes, ref_boxes):
    ref_tokens, tar_tokens, pair_geometry, geometry_embed, attn_grid = _common_tokens(
        head, xin, tar_boxes, ref_boxes
    )
    pair_tokens = torch.cat([ref_tokens, tar_tokens], dim=1)
    pair_tokens = head.token_norm(pair_tokens + geometry_embed.unsqueeze(1))

    scale_list = head.get_scale_list(head.scale_number).to(device=xin.device, dtype=xin.dtype)
    scale_queries = head.build_scale_queries(scale_list, geometry_embed)
    q = head.q_proj(head.query_norm(scale_queries))
    k = head.k_proj(pair_tokens)
    v = head.v_proj(pair_tokens)

    q = head.split_attention_heads(q)
    k = head.split_attention_heads(k)
    v = head.split_attention_heads(v)

    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(head.cross_attention_head_dim, 1))
    attn_logits = attn_logits * head.attn_logit_scale.exp().clamp(max=10.0)
    attn_heads = F.softmax(attn_logits, dim=-1)
    context = torch.matmul(attn_heads, v)
    context = head.merge_attention_heads(context)
    scale_states = head.context_norm(scale_queries + head.out_proj(context))
    logits = head.pred_head(scale_states).squeeze(-1)

    token_count = attn_grid * attn_grid
    attn = attn_heads.mean(dim=1)
    prob = F.softmax(logits, dim=-1)
    pred_scale = torch.sum(prob * scale_list.view(1, -1), dim=-1)
    return {
        "mode": "scale_query",
        "logits": logits,
        "prob": prob,
        "pred_scale": pred_scale,
        "scale_list": scale_list,
        "attn": attn,
        "attn_heads": attn_heads,
        "ref_attn": attn[..., :token_count],
        "tar_attn": attn[..., token_count:],
        "attn_grid": attn_grid,
        "pair_geometry": pair_geometry,
    }


@torch.no_grad()
def dense_qkv_debug(head, xin, tar_boxes, ref_boxes):
    _, H, W = xin.shape[-3:]
    scale_list = head.get_scale_list(head.scale_number).to(
        device=xin.device, dtype=xin.dtype
    )
    logits, local = head.dense_native_cross_attention_predict(
        xin, tar_boxes, ref_boxes, H, W, return_debug=True
    )
    if head.head_type == "distribution":
        prob = F.softmax(logits, dim=-1)
        pred_scale = torch.sum(prob * scale_list.view(1, -1), dim=-1)
    else:
        confidence = logits.sigmoid()
        topk = min(4, confidence.shape[-1])
        pred_conf, pred_bin = confidence.topk(topk, dim=-1)
        pred_conf = pred_conf / pred_conf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pred_scale = torch.sum(scale_list[pred_bin] * pred_conf, dim=-1)
        prob = confidence / confidence.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return {
        "mode": "dense_qkv",
        "logits": logits,
        "prob": prob,
        "pred_scale": pred_scale,
        "scale_list": scale_list,
        **local,
    }


@torch.no_grad()
def fp_sparse_qkv_debug(head, xin, tar_boxes, ref_boxes):
    _, H, W = xin.shape[-3:]
    scale_list = head.get_scale_list(head.scale_number).to(
        device=xin.device, dtype=xin.dtype
    )
    logits, local = head.fp_sparse_cross_attention_predict(
        xin, tar_boxes, ref_boxes, H, W, return_debug=True
    )
    if head.head_type == "distribution":
        prob = F.softmax(logits, dim=-1)
        pred_scale = torch.sum(prob * scale_list.view(1, -1), dim=-1)
    else:
        confidence = logits.sigmoid()
        topk = min(4, confidence.shape[-1])
        pred_conf, pred_bin = confidence.topk(topk, dim=-1)
        pred_conf = pred_conf / pred_conf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pred_scale = torch.sum(scale_list[pred_bin] * pred_conf, dim=-1)
        prob = confidence / confidence.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return {
        "mode": "fp_sparse_qkv",
        "logits": logits,
        "prob": prob,
        "pred_scale": pred_scale,
        "scale_list": scale_list,
        **local,
    }


@torch.no_grad()
def scale_match_debug(head, xin, tar_boxes, ref_boxes):
    _, H, W = xin.shape[-3:]
    scale_list = head.get_scale_list(head.scale_number).to(device=xin.device, dtype=xin.dtype)
    logits, local = head.scale_match_cross_attention_predict(
        xin, tar_boxes, ref_boxes, scale_list, H, W, return_debug=True
    )
    if head.head_type == "distribution":
        prob = F.softmax(logits, dim=-1)
        pred_scale = torch.sum(prob * scale_list.view(1, -1), dim=-1)
    else:
        confidence = logits.sigmoid()
        topk = min(4, confidence.shape[-1])
        pred_conf, pred_bin = confidence.topk(topk, dim=-1)
        pred_conf = pred_conf / pred_conf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pred_scale = torch.sum(scale_list[pred_bin] * pred_conf, dim=-1)
        prob = confidence / confidence.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return {
        "mode": "scale_match",
        "logits": logits,
        "prob": prob,
        "pred_scale": pred_scale,
        "scale_list": scale_list,
        **local,
    }


@torch.no_grad()
def cross_attention_debug(head, xin, tar_boxes, ref_boxes):
    mode = getattr(head, "cross_attention_mode", "scale_query")
    if mode == "dense_qkv":
        return dense_qkv_debug(head, xin, tar_boxes, ref_boxes)
    if mode == "fp_sparse_qkv":
        return fp_sparse_qkv_debug(head, xin, tar_boxes, ref_boxes)
    if mode == "scale_match":
        return scale_match_debug(head, xin, tar_boxes, ref_boxes)
    if mode == "ref_to_target":
        return ref_to_target_debug(head, xin, tar_boxes, ref_boxes)
    if mode == "dot_product":
        raise ValueError("dot_product mode has no cross-attention matrix to dump")
    return scale_query_debug(head, xin, tar_boxes, ref_boxes)


def save_heatmap_png(path, matrix, title, xlabel="token index", ylabel="query index"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(7.5, 4.5), dpi=160)
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def tensor_to_cpu(x):
    return x.detach().float().cpu().numpy()


def main():
    args = make_parser().parse_args()
    configure_module()
    os.makedirs(args.out_dir, exist_ok=True)

    exp = get_exp(args.exp_file, None)
    exp.merge(args.opts)
    exp.valset_dir = args.val_dir
    exp.valAnnoPath = args.val_dir
    exp.eval_batch_size = args.batch_size
    exp.data_num_workers = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = exp.get_model().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    if args.fp16:
        model.half()

    loader = exp.get_eval_loader(
        args.batch_size,
        is_distributed=False,
        data_path=args.val_dir,
        anno_path=args.val_dir,
    )

    summaries = []
    saved = 0
    for batch in loader:
        inps, dict_annos, boxes, ttc = batch
        if inps is None:
            continue
        inps = inps.to(device)
        boxes = boxes.to(device)
        ttc = ttc.to(device)
        if args.fp16:
            inps = inps.half()

        backbone_outs = model.backbone(inps)
        ref_boxes, tar_boxes = boxes[::2, :], boxes[1::2, :]
        debug = cross_attention_debug(model.head, backbone_outs, tar_boxes, ref_boxes)

        pred_scale = debug["pred_scale"]
        frame_gap = dict_annos.get("frame_gap", exp.sequence_len - 1)
        frame_gap = torch.as_tensor(frame_gap, device=device, dtype=pred_scale.dtype)
        fps = 10.0 / frame_gap
        pred_ttc = scale_ratio_to_ttc(pred_scale, fps=fps)
        gt_scale = ttc_to_scale_ratio(ttc.to(dtype=pred_scale.dtype), fps=fps)
        rte = torch.abs((pred_ttc - ttc) / (ttc + 1e-9)) * 100.0

        num_pairs = pred_scale.shape[0]
        for i in range(num_pairs):
            if saved >= args.num_samples:
                break
            sample_id = "%04d" % saved
            attn = tensor_to_cpu(debug["attn"][i])
            geom = tensor_to_cpu(debug["pair_geometry"][i])
            scale_np = tensor_to_cpu(debug["scale_list"])

            npz_path = os.path.join(args.out_dir, sample_id + ".npz")
            save_payload = {
                "mode": debug["mode"],
                "attn": attn,
                "attn_heads": tensor_to_cpu(debug["attn_heads"][i]),
                "scale_list": scale_np,
                "pair_geometry": geom,
                "pred_scale": float(pred_scale[i].detach().cpu()),
                "gt_scale": float(gt_scale[i].detach().cpu()),
                "pred_ttc": float(pred_ttc[i].detach().cpu()),
                "gt_ttc": float(ttc[i].detach().cpu()),
                "rte": float(rte[i].detach().cpu()),
                "attn_grid": int(debug.get("attn_grid", 0)),
            }
            if debug["mode"] in (
                    "dense_qkv", "fp_sparse_qkv", "scale_query", "scale_match"
            ):
                save_payload.update({
                    "logits": tensor_to_cpu(debug["logits"][i]),
                    "prob": tensor_to_cpu(debug["prob"][i]),
                })
                if debug["mode"] == "scale_query":
                    save_payload.update({
                        "ref_attn": tensor_to_cpu(debug["ref_attn"][i]),
                        "tar_attn": tensor_to_cpu(debug["tar_attn"][i]),
                    })
                elif debug["mode"] == "scale_match":
                    save_payload.update({
                        "baseline_scores": tensor_to_cpu(debug["baseline_scores"][i]),
                        "baseline_logits": tensor_to_cpu(debug["baseline_logits"][i]),
                        "attention_residual": tensor_to_cpu(debug["attention_residual"][i]),
                        "geometry_prior": tensor_to_cpu(debug["geometry_prior"][i]),
                        "attention_window": int(debug["attention_window"]),
                    })
                elif debug["mode"] == "dense_qkv":
                    save_payload.update({
                        "attention_window": int(debug["attention_window"]),
                        "native_hw": np.asarray(debug["native_hw"]),
                        "query_mask": tensor_to_cpu(debug["query_mask"][i]),
                        "reference_mask": tensor_to_cpu(debug["reference_mask"][i]),
                        "target_mask": tensor_to_cpu(debug["target_mask"][i]),
                        "expected_offset": tensor_to_cpu(debug["expected_offset"][i]),
                        "radial_displacement": tensor_to_cpu(
                            debug["radial_displacement"][i]
                        ),
                        "dilation_mass": tensor_to_cpu(debug["dilation_mass"][i]),
                        "attention_dilations": np.asarray(
                            debug["attention_dilations"]
                        ),
                        "attention_direction": debug["attention_direction"],
                    })
                else:
                    save_payload.update({
                        "native_hw": np.asarray(debug["native_hw"]),
                        "query_mask": tensor_to_cpu(debug["query_mask"][i]),
                        "reference_mask": tensor_to_cpu(debug["reference_mask"][i]),
                        "target_mask": tensor_to_cpu(debug["target_mask"][i]),
                        "expected_offset": tensor_to_cpu(debug["expected_offset"][i]),
                        "radial_displacement": tensor_to_cpu(
                            debug["radial_displacement"][i]
                        ),
                        "reliability_gate": tensor_to_cpu(
                            debug["reliability_gate"][i]
                        ),
                        "level_mass": tensor_to_cpu(debug["level_mass"][i]),
                        "sampling_offsets": tensor_to_cpu(
                            debug["sampling_offsets"][i]
                        ),
                        "attention_levels": int(debug["attention_levels"]),
                        "attention_points": int(debug["attention_points"]),
                        "attention_direction": debug["attention_direction"],
                        "attention_method": debug["attention_method"],
                    })
            else:
                save_payload.update({
                    "raw_scale": float(debug["raw_scale"][i].detach().cpu()),
                    "attn_entropy": float(debug["attn_entropy"][i].detach().cpu()),
                })
            np.savez_compressed(npz_path, **save_payload)

            meta = {
                "sample": sample_id,
                "mode": debug["mode"],
                "npz": npz_path,
                "pred_scale": float(pred_scale[i].detach().cpu()),
                "gt_scale": float(gt_scale[i].detach().cpu()),
                "pred_ttc": float(pred_ttc[i].detach().cpu()),
                "gt_ttc": float(ttc[i].detach().cpu()),
                "rte": float(rte[i].detach().cpu()),
            }
            if debug["mode"] in (
                    "dense_qkv", "fp_sparse_qkv", "scale_query", "scale_match"
            ):
                prob = debug["prob"]
                scale_list = debug["scale_list"]
                meta.update({
                    "peak_scale": float(scale_list[prob[i].argmax()].detach().cpu()),
                    "peak_prob": float(prob[i].max().detach().cpu()),
                    "entropy": float((-(prob[i] * (prob[i] + 1e-12).log()).sum()).detach().cpu()),
                })
                if debug["mode"] == "scale_query":
                    meta.update({
                        "ref_mass_mean": float(debug["ref_attn"][i].sum(dim=-1).mean().detach().cpu()),
                        "tar_mass_mean": float(debug["tar_attn"][i].sum(dim=-1).mean().detach().cpu()),
                    })
                    title = "scale query attention, sample " + sample_id
                    xlabel = "ref+target token index"
                elif debug["mode"] == "scale_match":
                    meta.update({
                        "attn_entropy": float(debug["attn_entropy"][i].detach().cpu()),
                        "attention_window": int(debug["attention_window"]),
                    })
                    title = "local scale-match attention, sample " + sample_id
                    xlabel = "local offset index"
                elif debug["mode"] == "dense_qkv":
                    meta.update({
                        "attn_entropy": float(debug["attn_entropy"][i].detach().cpu()),
                        "attention_window": int(debug["attention_window"]),
                        "attention_dilations": list(debug["attention_dilations"]),
                        "dilation_mass": tensor_to_cpu(
                            debug["dilation_mass"][i]
                        ).tolist(),
                        "native_hw": list(debug["native_hw"]),
                        "attention_direction": debug["attention_direction"],
                    })
                    title = "native target-Q/reference-KV attention, sample " + sample_id
                    xlabel = "local offset index"
                else:
                    meta.update({
                        "attn_entropy": float(debug["attn_entropy"][i].detach().cpu()),
                        "attention_levels": int(debug["attention_levels"]),
                        "attention_points": int(debug["attention_points"]),
                        "level_mass": tensor_to_cpu(
                            debug["level_mass"][i]
                        ).tolist(),
                        "reliability_mean": float(
                            debug["reliability_gate"][i].mean().detach().cpu()
                        ),
                        "native_hw": list(debug["native_hw"]),
                        "attention_direction": debug["attention_direction"],
                    })
                    title = "FP-sparse target-Q/reference-KV attention, sample " + sample_id
                    xlabel = "sparse sample index"
                save_heatmap_png(
                    os.path.join(args.out_dir, sample_id + "_all_attn.png"),
                    attn,
                    title,
                    xlabel=xlabel,
                    ylabel=(
                        "aggregate"
                        if debug["mode"] in ("dense_qkv", "fp_sparse_qkv")
                        else "scale bin"
                    ),
                )
            else:
                meta.update({
                    "attn_entropy": float(debug["attn_entropy"][i].detach().cpu()),
                })
                save_heatmap_png(
                    os.path.join(args.out_dir, sample_id + "_ref_to_target_attn.png"),
                    attn,
                    "ref-to-target attention, sample " + sample_id,
                    xlabel="target token index",
                    ylabel="ref token index",
                )
            summaries.append(meta)
            saved += 1

        if saved >= args.num_samples:
            break

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print("Saved %d samples to %s" % (saved, args.out_dir))
    print("Summary: %s" % summary_path)


if __name__ == "__main__":
    main()
