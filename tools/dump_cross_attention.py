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


@torch.no_grad()
def cross_attention_debug(head, xin, tar_boxes, ref_boxes):
    C, H, W = xin.shape[-3:]
    scale_list = head.get_scale_list(head.scale_number).to(device=xin.device, dtype=xin.dtype)
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
    pair_tokens = torch.cat([ref_tokens, tar_tokens], dim=1)
    pair_tokens = head.token_norm(pair_tokens + geometry_embed.unsqueeze(1))

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
    ref_attn = attn[..., :token_count]
    tar_attn = attn[..., token_count:]
    return {
        "logits": logits,
        "prob": F.softmax(logits, dim=-1),
        "scale_list": scale_list,
        "attn": attn,
        "attn_heads": attn_heads,
        "ref_attn": ref_attn,
        "tar_attn": tar_attn,
        "attn_grid": attn_grid,
        "pair_geometry": pair_geometry,
    }


def save_heatmap_png(path, matrix, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(7.5, 4.5), dpi=160)
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.xlabel("token index")
    plt.ylabel("scale bin")
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

        prob = debug["prob"]
        scale_list = debug["scale_list"]
        pred_scale = (prob * scale_list.view(1, -1)).sum(dim=-1)
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
            ref_attn = tensor_to_cpu(debug["ref_attn"][i])
            tar_attn = tensor_to_cpu(debug["tar_attn"][i])
            logits = tensor_to_cpu(debug["logits"][i])
            prob_i = tensor_to_cpu(prob[i])
            scale_np = tensor_to_cpu(scale_list)
            geom = tensor_to_cpu(debug["pair_geometry"][i])

            npz_path = os.path.join(args.out_dir, sample_id + ".npz")
            np.savez_compressed(
                npz_path,
                attn=attn,
                attn_heads=tensor_to_cpu(debug["attn_heads"][i]),
                ref_attn=ref_attn,
                tar_attn=tar_attn,
                logits=logits,
                prob=prob_i,
                scale_list=scale_np,
                pair_geometry=geom,
                pred_scale=float(pred_scale[i].detach().cpu()),
                gt_scale=float(gt_scale[i].detach().cpu()),
                pred_ttc=float(pred_ttc[i].detach().cpu()),
                gt_ttc=float(ttc[i].detach().cpu()),
                rte=float(rte[i].detach().cpu()),
                attn_grid=int(debug["attn_grid"]),
            )

            meta = {
                "sample": sample_id,
                "npz": npz_path,
                "pred_scale": float(pred_scale[i].detach().cpu()),
                "gt_scale": float(gt_scale[i].detach().cpu()),
                "pred_ttc": float(pred_ttc[i].detach().cpu()),
                "gt_ttc": float(ttc[i].detach().cpu()),
                "rte": float(rte[i].detach().cpu()),
                "peak_scale": float(scale_list[prob[i].argmax()].detach().cpu()),
                "peak_prob": float(prob[i].max().detach().cpu()),
                "entropy": float((-(prob[i] * (prob[i] + 1e-12).log()).sum()).detach().cpu()),
                "ref_mass_mean": float(debug["ref_attn"][i].sum(dim=-1).mean().detach().cpu()),
                "tar_mass_mean": float(debug["tar_attn"][i].sum(dim=-1).mean().detach().cpu()),
            }
            summaries.append(meta)

            save_heatmap_png(
                os.path.join(args.out_dir, sample_id + "_all_attn.png"),
                attn,
                "scale query attention, sample " + sample_id,
            )
            save_heatmap_png(
                os.path.join(args.out_dir, sample_id + "_ref_attn.png"),
                ref_attn,
                "ref-token attention, sample " + sample_id,
            )
            save_heatmap_png(
                os.path.join(args.out_dir, sample_id + "_tar_attn.png"),
                tar_attn,
                "target-token attention, sample " + sample_id,
            )
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
