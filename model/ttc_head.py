from torch import nn
from .network_blocks import FocalLoss
from torchvision.ops import roi_align
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from data.ttc_dataset import ttc_to_scale_ratio, scale_ratio_to_ttc


class TTCHead(nn.Module):
    def __init__(
            self,
            scale_number=10,
            in_channel=24,
            fps=None,
            ttc_bin=False,
            min_scale=0.8,
            max_scale=1.2,
            shift = False,
            shift_kernel_size = 3,
            grid_size = -1,
            use_focal_loss = False,
            normed_box = False,
            sequence_len = 6,
            smoother_factor = 1.0,
            head_type = "bce",
            normalize_similarity = True,
            similarity_topk_ratio = 0.05,
            similarity_topk_weight = 0.4,
            use_cross_attention = True,
            cross_attention_grid_size = 16,
            cross_attention_position_sigma = 0.35,
            cross_attention_dim = None,
            cross_attention_heads = 4,
            cross_attention_mode = "dense_qkv",
            cross_attention_window_size = 3,
            cross_attention_dilations = (1, 3, 6),
            cross_attention_dropout = 0.0,
            dense_attention_context_scale = 1.0,
            dense_attention_align_centers = True,
            cross_attention_residual_init = 0.0,
            cross_attention_geometry_sigma = 0.08,
            cross_attention_geometry_weight = 0.0,
            cross_attention_reg_loss_weight = 0.25,
            cross_attention_ttc_loss_weight = 1.0,
            ttc_metric_clip = 20.0,
            ttc_metric_min_denom = 1.0,
            ttc_metric_huber_beta = 0.1,
            ttc_metric_short_loss_weight = 0.1,
            ttc_metric_mid_ttc_abs_thresh = 3.0,
            ttc_metric_mid_loss_weight = 0.5,
            ttc_metric_long_ttc_abs_thresh = 6.0,
            ttc_metric_long_loss_weight = 0.85,
            ttc_metric_tail_ttc_abs_thresh = 12.0,
            ttc_metric_tail_loss_weight = 1.0,
            **kwargs
    ):
        super().__init__()
        #scale_number = scale_bin + 1
        if use_focal_loss:
            self.loss = FocalLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.scale_number = scale_number
        self.in_channel = in_channel
        self.ttc_bin = ttc_bin
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.total_stride = 1
        self.shift = shift
        self.shift_kernel_size = shift_kernel_size
        self.grid_size = grid_size
        self.fps = fps
        self.sequence_len = sequence_len
        self.normed_box = normed_box
        self.smoother_factor = smoother_factor # training only
        self.head_type = head_type
        self.normalize_similarity = normalize_similarity
        self.similarity_topk_ratio = similarity_topk_ratio
        self.similarity_topk_weight = similarity_topk_weight
        self.use_cross_attention = bool(use_cross_attention)
        if not self.use_cross_attention:
            cross_attention_mode = "dot_product"
        self.cross_attention_grid_size = cross_attention_grid_size
        self.cross_attention_position_sigma = cross_attention_position_sigma
        self.cross_attention_mode = str(cross_attention_mode).lower()
        if self.cross_attention_mode not in (
                "dense_qkv", "scale_match", "ref_to_target", "scale_query", "dot_product"
        ):
            raise ValueError(
                "cross_attention_mode must be 'dense_qkv', 'scale_match', "
                "'ref_to_target', 'scale_query', or 'dot_product'."
            )
        cross_attention_window_size = int(cross_attention_window_size)
        if cross_attention_window_size < 1 or cross_attention_window_size % 2 == 0:
            raise ValueError("cross_attention_window_size must be a positive odd number.")
        self.cross_attention_window_size = cross_attention_window_size
        self.cross_attention_dropout = float(cross_attention_dropout)
        if not 0.0 <= self.cross_attention_dropout < 1.0:
            raise ValueError("cross_attention_dropout must be in [0, 1).")
        if isinstance(cross_attention_dilations, int):
            cross_attention_dilations = (cross_attention_dilations,)
        self.cross_attention_dilations = tuple(
            dict.fromkeys(int(value) for value in cross_attention_dilations)
        )
        if not self.cross_attention_dilations or any(
                value < 1 for value in self.cross_attention_dilations
        ):
            raise ValueError("cross_attention_dilations must contain positive integers.")
        self.dense_attention_context_scale = max(float(dense_attention_context_scale), 1.0)
        self.dense_attention_align_centers = bool(dense_attention_align_centers)
        self.cross_attention_geometry_sigma = max(float(cross_attention_geometry_sigma), 1e-3)
        self.cross_attention_reg_loss_weight = max(float(cross_attention_reg_loss_weight), 0.0)
        self.cross_attention_ttc_loss_weight = max(float(cross_attention_ttc_loss_weight), 0.0)
        self.ttc_metric_clip = float(ttc_metric_clip)
        self.ttc_metric_min_denom = max(float(ttc_metric_min_denom), 1e-6)
        self.ttc_metric_huber_beta = max(float(ttc_metric_huber_beta), 1e-6)
        self.ttc_metric_short_loss_weight = float(ttc_metric_short_loss_weight)
        self.ttc_metric_mid_ttc_abs_thresh = float(ttc_metric_mid_ttc_abs_thresh)
        self.ttc_metric_mid_loss_weight = float(ttc_metric_mid_loss_weight)
        self.ttc_metric_long_ttc_abs_thresh = float(ttc_metric_long_ttc_abs_thresh)
        self.ttc_metric_long_loss_weight = float(ttc_metric_long_loss_weight)
        self.ttc_metric_tail_ttc_abs_thresh = float(ttc_metric_tail_ttc_abs_thresh)
        self.ttc_metric_tail_loss_weight = float(ttc_metric_tail_loss_weight)
        self.ce_loss = nn.CrossEntropyLoss()
        self.register_buffer(
            "_default_scale_list",
            torch.linspace(self.min_scale, self.max_scale, self.scale_number),
            persistent=False,
        )

        attn_dim = cross_attention_dim or in_channel
        cross_attention_heads = int(cross_attention_heads)
        if cross_attention_heads < 1:
            raise ValueError("cross_attention_heads must be >= 1.")
        if attn_dim % cross_attention_heads != 0:
            raise ValueError(
                "cross_attention_dim/in_channel (%d) must be divisible by cross_attention_heads (%d)."
                % (attn_dim, cross_attention_heads)
            )
        self.attn_dim = attn_dim
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_head_dim = attn_dim // cross_attention_heads
        # Keep the official calibration layer name so old dot-product weights can
        # warm-start the exact fallback branch of the hybrid matcher.
        self.scale_preds = nn.Linear(scale_number, scale_number)
        with torch.no_grad():
            self.scale_preds.weight.copy_(torch.eye(scale_number))
            self.scale_preds.bias.zero_()
        self.ref_proj = nn.Linear(in_channel, attn_dim)
        self.tar_proj = nn.Linear(in_channel, attn_dim)
        self.q_proj = nn.Linear(attn_dim, attn_dim)
        self.k_proj = nn.Linear(attn_dim, attn_dim)
        self.v_proj = nn.Linear(attn_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, attn_dim)
        self.token_type_embed = nn.Embedding(2, attn_dim)
        self.position_proj = nn.Linear(2, attn_dim)
        self.scale_index_embed = nn.Embedding(scale_number, attn_dim)
        self.scale_value_proj = nn.Sequential(
            nn.Linear(1, attn_dim),
            nn.SiLU(inplace=True),
            nn.Linear(attn_dim, attn_dim),
        )
        self.geometry_proj = nn.Sequential(
            nn.Linear(8, attn_dim),
            nn.SiLU(inplace=True),
            nn.Linear(attn_dim, attn_dim),
        )
        self.token_norm = nn.LayerNorm(attn_dim)
        self.query_norm = nn.LayerNorm(attn_dim)
        self.context_norm = nn.LayerNorm(attn_dim)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, attn_dim),
            nn.SiLU(inplace=True),
            nn.Linear(attn_dim, 1),
        )
        self.scale_regression_head = nn.Sequential(
            nn.LayerNorm(attn_dim * 6),
            nn.Linear(attn_dim * 6, attn_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(attn_dim * 2, 1),
        )
        self.attn_logit_scale = nn.Parameter(torch.zeros(1))
        self.local_attn_logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.dilation_bias = nn.Parameter(
            torch.linspace(0.0, -0.2, len(self.cross_attention_dilations))
        )
        self.match_feature_norm = nn.LayerNorm(8)
        match_hidden = max(attn_dim, 16)
        self.match_scale_mixer = nn.Sequential(
            nn.Conv1d(8, match_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(match_hidden, match_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(match_hidden, 1, kernel_size=1),
        )
        nn.init.normal_(self.match_scale_mixer[-1].weight, std=2e-2)
        nn.init.zeros_(self.match_scale_mixer[-1].bias)
        self.attention_residual_gate = nn.Parameter(torch.tensor(float(cross_attention_residual_init)))
        # Zero initialization preserves the official dot-product logits at
        # cold start; data must earn any direct geometry contribution.
        self.geometry_prior_weight = nn.Parameter(
            torch.tensor(float(cross_attention_geometry_weight))
        )
        # Five confidence statistics, six explicit displacement statistics, and
        # one attention-mass statistic per dilation preserve the scale-motion
        # signal that would otherwise disappear during global aggregation.
        dense_feature_dim = (
            attn_dim * 6 + 11 + len(self.cross_attention_dilations)
        )
        self.dense_scale_head = nn.Sequential(
            nn.LayerNorm(dense_feature_dim),
            nn.Linear(dense_feature_dim, attn_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(self.cross_attention_dropout),
            nn.Linear(attn_dim * 2, attn_dim),
            nn.SiLU(inplace=True),
            nn.Linear(attn_dim, scale_number),
        )
        self.freeze_inactive_dense_branches()

    @staticmethod
    def set_module_trainable(module, trainable):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def freeze_inactive_dense_branches(self):
        """Avoid DDP reduction failures from legacy branches unused by dense_qkv."""
        if self.cross_attention_mode != "dense_qkv":
            return
        for module in (
                self.scale_preds,
                self.scale_index_embed,
                self.scale_value_proj,
                self.pred_head,
                self.scale_regression_head,
                self.match_feature_norm,
                self.match_scale_mixer,
        ):
            self.set_module_trainable(module, False)
        for parameter in (
                self.attn_logit_scale,
                self.attention_residual_gate,
                self.geometry_prior_weight,
        ):
            parameter.requires_grad_(False)

    def forward(self, xin, tar_boxes,ref_boxes,ttc_imu =None,**kwargs):
        '''

        :param xin: input features from backbone
        :param tar_boxes: input boxes in [batch_idx, x1, y1, x2, y2] format #[K,5]
        :param ref_boxes:
        :return: predicted scale confidences
        '''

        C,H,W = xin.shape[-3:]
        S = self.scale_number
        scale_list = self.get_scale_list(S)
        if self.cross_attention_mode == "ref_to_target":
            pred_scales = self.ref_target_cross_attention_predict(xin, tar_boxes, ref_boxes, H, W)
            if self.training:
                if 'dictAnnos' in kwargs:
                    frame_gap = kwargs['dictAnnos']['frame_gap']
                else:
                    frame_gap = self.sequence_len - 1
                gt_scales = self.prepare_targets(ttc_imu, frame_gap, pred_scales)
                gt_scales = gt_scales.clamp(self.min_scale, self.max_scale)
                return F.l1_loss(pred_scales, gt_scales, reduction="mean")
            return pred_scales.unsqueeze(-1), scale_list, pred_scales

        if self.cross_attention_mode == "dense_qkv":
            predictions = self.dense_native_cross_attention_predict(
                xin, tar_boxes, ref_boxes, H, W
            )
        elif self.cross_attention_mode == "scale_match":
            predictions = self.scale_match_cross_attention_predict(
                xin, tar_boxes, ref_boxes, scale_list, H, W
            )
        elif self.cross_attention_mode == "dot_product":
            predictions, _, _ = self.legacy_dot_product_scores(
                xin, tar_boxes, ref_boxes, scale_list, H, W
            )
            predictions = self.scale_preds(predictions)
        else:
            predictions = self.cross_attention_predict(xin, tar_boxes, ref_boxes, scale_list, H, W)
        if self.training:
            if self.head_type == "distribution":
                if 'dictAnnos' in kwargs:
                    frame_gap = kwargs['dictAnnos']['frame_gap']
                else:
                    frame_gap = self.sequence_len - 1
                gt_scales = self.prepare_targets(ttc_imu, frame_gap, predictions)
                if self.cross_attention_mode in ("dense_qkv", "scale_match"):
                    return self.get_scale_match_loss(
                        predictions, gt_scales, scale_list, ttc_imu, frame_gap
                    )
                return self.get_distribution_loss(predictions, gt_scales, scale_list)

            predictions = predictions.view(-1,1)
            if 'dictAnnos' in kwargs:
                dictAnnos = kwargs['dictAnnos']
                frame_gap = dictAnnos['frame_gap']
            else:
                frame_gap = self.sequence_len-1
            gt_one_hot = self.gt_to_one_hot(ttc_imu, gap=frame_gap,scale_list=scale_list, ref_tensor=predictions)
            scale_loss = self.get_loss(predictions, gt_one_hot)
            return scale_loss
        if self.head_type == "distribution":
            return F.softmax(predictions, dim=-1), scale_list, None
        return predictions.sigmoid(), scale_list, None

    def dense_native_cross_attention_predict(
            self, xin, tar_boxes, ref_boxes, H, W, return_debug=False
    ):
        """Native-resolution target-Q/reference-KV sliding cross-attention.

        This path never calls ROI Align and never resizes the target feature
        map.  Reference features are translated (not scaled) so the two box
        centres share a coordinate system; the inter-frame object-size change
        therefore remains visible to the attention head.
        """
        if xin.ndim != 4 or xin.shape[0] % 2 != 0:
            raise ValueError("dense_qkv expects an even NCHW tensor of interleaved frame pairs.")
        if tar_boxes.ndim != 2 or ref_boxes.ndim != 2:
            raise ValueError("tar_boxes and ref_boxes must be rank-2 tensors.")
        if tar_boxes.shape != ref_boxes.shape or tar_boxes.shape[-1] != 5:
            raise ValueError("tar_boxes and ref_boxes must have the same [objects, 5] shape.")
        if tar_boxes.shape[0] == 0:
            raise ValueError("dense_qkv requires at least one reference/target box pair.")
        ref_maps_all = xin[::2]
        tar_maps_all = xin[1::2]
        ref_indices = ref_boxes[:, 0].long()
        tar_indices = tar_boxes[:, 0].long()
        if (
                (ref_indices < 0).any()
                or (ref_indices >= ref_maps_all.shape[0]).any()
                or (tar_indices < 0).any()
                or (tar_indices >= tar_maps_all.shape[0]).any()
        ):
            raise IndexError("A dense_qkv box batch index is outside the available frame pairs.")
        ref_maps = ref_maps_all.index_select(0, ref_indices)
        tar_maps = tar_maps_all.index_select(0, tar_indices)
        batch = tar_maps.shape[0]
        tokens = H * W
        window = self.cross_attention_window_size

        pair_geometry = self.box_pair_geometry(ref_boxes, tar_boxes, H, W).type_as(xin)
        geometry_embed = self.geometry_proj(pair_geometry)
        ref_position = self.box_relative_position_tokens(
            ref_boxes, H, W, xin.dtype, xin.device
        )
        tar_position = self.box_relative_position_tokens(
            tar_boxes, H, W, xin.dtype, xin.device
        )

        ref_raw = ref_maps.permute(0, 2, 3, 1).reshape(batch, tokens, self.in_channel)
        tar_raw = tar_maps.permute(0, 2, 3, 1).reshape(batch, tokens, self.in_channel)
        ref_tokens = self.ref_proj(ref_raw)
        ref_tokens = self.token_norm(
            ref_tokens
            + self.token_type_embed.weight[0].view(1, 1, -1)
            + self.position_proj(ref_position)
        )
        tar_tokens = self.tar_proj(tar_raw)
        tar_tokens = self.token_norm(
            tar_tokens
            + self.token_type_embed.weight[1].view(1, 1, -1)
            + self.position_proj(tar_position)
        )

        ref_state_map = ref_tokens.transpose(1, 2).reshape(
            batch, self.attn_dim, H, W
        )
        ref_mask = self.native_box_mask(
            ref_boxes, H, W, self.dense_attention_context_scale, xin.dtype, xin.device
        )
        tar_mask = self.native_box_mask(
            tar_boxes, H, W, self.dense_attention_context_scale, xin.dtype, xin.device
        )
        if self.dense_attention_align_centers:
            ref_state_map = self.translate_reference_to_target(
                ref_state_map, ref_boxes, tar_boxes, H, W
            )
            ref_mask = self.translate_reference_to_target(
                ref_mask, ref_boxes, tar_boxes, H, W
            )
        aligned_ref_tokens = ref_state_map.flatten(start_dim=-2).transpose(1, 2)

        # The current/target frame is Q; the centre-aligned reference frame is
        # K/V. This keeps the output anchored to the frame whose TTC is being
        # estimated, while retaining the reference object's native scale.
        q = self.split_attention_heads(
            self.q_proj(self.query_norm(tar_tokens))
        )
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k_map = self.k_proj(aligned_ref_tokens).transpose(1, 2).reshape(
            batch, self.attn_dim, H, W
        )
        v_map = self.v_proj(aligned_ref_tokens).transpose(1, 2).reshape(
            batch, self.attn_dim, H, W
        )

        key_masks = []
        position_biases = []
        key_offsets = []
        dilation_slices = []
        key_start = 0
        for dilation_index, dilation in enumerate(self.cross_attention_dilations):
            mask_patch = F.unfold(
                ref_mask,
                kernel_size=window,
                dilation=dilation,
                padding=(window // 2) * dilation,
            ).transpose(1, 2)
            branch_keys = window * window
            key_masks.append(mask_patch)
            position_biases.append(
                self.local_position_bias(window, xin.device, xin.dtype)
                + self.dilation_bias[dilation_index].to(dtype=xin.dtype)
            )
            key_offsets.append(
                self.local_key_offsets(window, dilation, xin.device, xin.dtype)
            )
            dilation_slices.append(slice(key_start, key_start + branch_keys))
            key_start += branch_keys

        key_mask = torch.cat(key_masks, dim=-1)
        valid_keys = key_mask[:, None] > 1e-4
        has_valid_key = valid_keys.any(dim=-1).squeeze(1)
        query_mask = tar_mask.flatten(start_dim=1) * has_valid_key.to(xin.dtype)
        position_bias = torch.cat(position_biases, dim=-1)
        attention_args = (
            q,
            k_map,
            v_map,
            valid_keys,
            query_mask,
            position_bias,
            self.local_attn_logit_scale,
        )
        if self.training and torch.is_grad_enabled():
            attention_weights, context = checkpoint(
                self.native_multi_dilation_attention,
                *attention_args,
                use_reentrant=False,
            )
        else:
            attention_weights, context = self.native_multi_dilation_attention(
                *attention_args
            )
        context = self.out_proj(self.merge_attention_heads(context))
        match_tokens = self.context_norm(tar_tokens + context)

        tar_mean = self.masked_token_mean(
            tar_tokens, query_mask
        )
        ref_mean = self.masked_token_mean(
            aligned_ref_tokens, ref_mask.flatten(start_dim=1)
        )
        context_mean = self.masked_token_mean(context, query_mask)
        delta_mean = self.masked_token_mean(
            (match_tokens - tar_tokens).abs(), query_mask
        )
        product_mean = self.masked_token_mean(
            tar_tokens * context, query_mask
        )

        aligned_similarity = F.cosine_similarity(
            tar_tokens, context, dim=-1, eps=1e-6
        )
        similarity_mean = self.masked_scalar_mean(
            aligned_similarity, query_mask
        )
        similarity_std = self.masked_scalar_std(
            aligned_similarity, query_mask, similarity_mean
        )
        similarity_max = self.masked_scalar_max(
            aligned_similarity, query_mask
        )
        attention_confidence = self.masked_scalar_mean(
            attention_weights.max(dim=-1).values.mean(dim=1), query_mask
        )
        attention_entropy = -(
            attention_weights * (attention_weights + 1e-12).log()
        ).sum(dim=-1)
        attention_entropy = attention_entropy.mean(dim=1)
        total_keys = attention_weights.shape[-1]
        attention_focus = 1.0 - self.masked_scalar_mean(
            attention_entropy, query_mask
        ) / math.log(max(total_keys, 2))
        scalar_features = torch.stack([
            similarity_mean,
            similarity_max,
            similarity_std,
            attention_confidence,
            attention_focus,
        ], dim=-1)

        offsets = torch.cat(key_offsets, dim=0)
        expected_offsets_per_head = (
            attention_weights.unsqueeze(-1)
            * offsets.view(1, 1, 1, total_keys, 2)
        ).sum(dim=-2)
        expected_offsets = expected_offsets_per_head.mean(dim=1)
        tar_xyxy = self.boxes_to_feature_xyxy(tar_boxes, H, W).type_as(xin)
        tar_half_size = torch.stack([
            (tar_xyxy[:, 2] - tar_xyxy[:, 0]).clamp_min(1.0) * 0.5,
            (tar_xyxy[:, 3] - tar_xyxy[:, 1]).clamp_min(1.0) * 0.5,
        ], dim=-1)
        normalized_offsets = expected_offsets / tar_half_size[:, None, :]
        radial_unit = tar_position / tar_position.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        tangential_unit = torch.stack([
            -radial_unit[..., 1], radial_unit[..., 0]
        ], dim=-1)
        radial_displacement = (normalized_offsets * radial_unit).sum(dim=-1)
        tangential_displacement = (
            normalized_offsets * tangential_unit
        ).sum(dim=-1)
        offset_magnitude = normalized_offsets.norm(dim=-1)
        radial_mean = self.masked_scalar_mean(radial_displacement, query_mask)
        motion_features = torch.stack([
            radial_mean,
            self.masked_scalar_std(
                radial_displacement, query_mask, radial_mean
            ),
            self.masked_scalar_mean(radial_displacement.abs(), query_mask),
            self.masked_scalar_mean(offset_magnitude, query_mask),
            self.masked_scalar_std(
                offset_magnitude,
                query_mask,
                self.masked_scalar_mean(offset_magnitude, query_mask),
            ),
            self.masked_scalar_mean(
                tangential_displacement.abs(), query_mask
            ),
        ], dim=-1)
        dilation_mass = torch.stack([
            self.masked_scalar_mean(
                attention_weights[..., dilation_slice].sum(dim=-1).mean(dim=1),
                query_mask,
            )
            for dilation_slice in dilation_slices
        ], dim=-1)
        pair_feature = torch.cat([
            tar_mean,
            ref_mean,
            context_mean,
            delta_mean,
            product_mean,
            geometry_embed,
            scalar_features,
            motion_features,
            dilation_mass,
        ], dim=-1)
        predictions = self.dense_scale_head(pair_feature)
        if not return_debug:
            return predictions

        weighted_attn = attention_weights * query_mask[:, None, :, None]
        attn_summary = weighted_attn.sum(dim=(1, 2)) / (
            query_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            * self.cross_attention_heads
        )
        debug = {
            "attn": attn_summary.unsqueeze(1),
            "attn_heads": attention_weights,
            "attn_entropy": self.masked_scalar_mean(
                attention_entropy, query_mask
            ),
            "pair_geometry": pair_geometry,
            "query_mask": query_mask.view(batch, 1, H, W),
            "reference_mask": ref_mask,
            "target_mask": tar_mask,
            "expected_offset": expected_offsets.transpose(1, 2).reshape(
                batch, 2, H, W
            ),
            "radial_displacement": radial_displacement.view(batch, 1, H, W),
            "dilation_mass": dilation_mass,
            "native_hw": (H, W),
            "attention_window": window,
            "attention_dilations": self.cross_attention_dilations,
            "attention_direction": "target_q_reference_kv",
        }
        return predictions, debug

    def native_box_mask(self, boxes, H, W, context_scale, dtype, device):
        box_xyxy = self.boxes_to_feature_xyxy(
            boxes.to(device=device, dtype=dtype), H, W
        )
        x1, y1, x2, y2 = box_xyxy.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half_w = (x2 - x1).clamp_min(1.0) * context_scale * 0.5
        half_h = (y2 - y1).clamp_min(1.0) * context_scale * 0.5
        xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, W) + 0.5
        yy = torch.arange(H, device=device, dtype=dtype).view(1, H, 1) + 0.5
        mask_x = (xx >= (cx - half_w)[:, None, None]) & (
            xx <= (cx + half_w)[:, None, None]
        )
        mask_y = (yy >= (cy - half_h)[:, None, None]) & (
            yy <= (cy + half_h)[:, None, None]
        )
        return (mask_x & mask_y).to(dtype=dtype).unsqueeze(1)

    def box_relative_position_tokens(self, boxes, H, W, dtype, device):
        box_xyxy = self.boxes_to_feature_xyxy(
            boxes.to(device=device, dtype=dtype), H, W
        )
        x1, y1, x2, y2 = box_xyxy.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half_w = (x2 - x1).clamp_min(1.0) * 0.5
        half_h = (y2 - y1).clamp_min(1.0) * 0.5
        xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, W) + 0.5
        yy = torch.arange(H, device=device, dtype=dtype).view(1, H, 1) + 0.5
        rel_x = ((xx - cx[:, None, None]) / half_w[:, None, None]).expand(-1, H, -1)
        rel_y = ((yy - cy[:, None, None]) / half_h[:, None, None]).expand(-1, -1, W)
        return torch.stack([rel_x, rel_y], dim=-1).clamp(-4.0, 4.0).reshape(
            boxes.shape[0], H * W, 2
        )

    def translate_reference_to_target(
            self, tensor, ref_boxes, tar_boxes, H, W
    ):
        ref = self.boxes_to_feature_xyxy(ref_boxes, H, W).type_as(tensor)
        tar = self.boxes_to_feature_xyxy(tar_boxes, H, W).type_as(tensor)
        ref_cx = (ref[:, 0] + ref[:, 2]) * 0.5
        ref_cy = (ref[:, 1] + ref[:, 3]) * 0.5
        tar_cx = (tar[:, 0] + tar[:, 2]) * 0.5
        tar_cy = (tar[:, 1] + tar[:, 3]) * 0.5
        x = (torch.arange(W, device=tensor.device, dtype=tensor.dtype) + 0.5) * 2.0 / W - 1.0
        y = (torch.arange(H, device=tensor.device, dtype=tensor.dtype) + 0.5) * 2.0 / H - 1.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(
            tensor.shape[0], -1, -1, -1
        ).clone()
        grid[..., 0] += (2.0 * (ref_cx - tar_cx) / W)[:, None, None]
        grid[..., 1] += (2.0 * (ref_cy - tar_cy) / H)[:, None, None]
        return F.grid_sample(
            tensor,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def native_multi_dilation_attention(
            self,
            q,
            k_map,
            v_map,
            valid_keys,
            query_mask,
            position_bias,
            logit_scale,
    ):
        """Compute dilation branches sequentially to cap native-resolution memory."""
        window = self.cross_attention_window_size
        logits = []
        for dilation in self.cross_attention_dilations:
            key_patch = self.unfold_native_local_heads(
                k_map, window, dilation=dilation
            )
            key_patch = F.normalize(key_patch, p=2, dim=-1, eps=1e-6)
            # Use batched matrix multiplication instead of materialising the
            # broadcast product [B, heads, tokens, keys, head_dim].  The latter
            # can require several GiB for native-resolution crops even though
            # the reduced logits are small.
            logits.append(
                torch.matmul(
                    q.unsqueeze(-2), key_patch.transpose(-1, -2)
                ).squeeze(-2)
            )
        attention_logits = torch.cat(logits, dim=-1)
        attention_logits = (
            attention_logits * logit_scale.exp().clamp(1.0, 20.0)
            + position_bias
        )
        attention_logits = attention_logits.masked_fill(~valid_keys, -1e4)
        attention_weights = F.softmax(
            attention_logits.float(), dim=-1
        ).to(attention_logits.dtype)
        # Softmax over an entirely masked window is otherwise uniform. Remove
        # invalid keys explicitly and renormalize only windows containing data.
        attention_weights = (
            attention_weights * valid_keys.to(attention_weights.dtype)
        )
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        attention_weights = attention_weights * (
            query_mask[:, None, :, None] > 1e-4
        ).to(attention_weights.dtype)
        dropped_attention = F.dropout(
            attention_weights,
            p=self.cross_attention_dropout,
            training=self.training,
        )

        context = torch.zeros_like(q)
        branch_keys = window * window
        for index, dilation in enumerate(self.cross_attention_dilations):
            value_patch = self.unfold_native_local_heads(
                v_map, window, dilation=dilation
            )
            start = index * branch_keys
            branch_attention = dropped_attention[
                ..., start:start + branch_keys
            ]
            # As above, avoid a keys-by-head_dim broadcast temporary during
            # value aggregation.  This is algebraically identical to the
            # elementwise product followed by sum over the key dimension.
            branch_context = torch.matmul(
                branch_attention.unsqueeze(-2), value_patch
            ).squeeze(-2)
            context = context + branch_context
        return attention_weights, context

    def unfold_native_local_heads(self, feature_map, window, dilation=1):
        batch, _, _, _ = feature_map.shape
        patches = F.unfold(
            feature_map,
            kernel_size=window,
            dilation=dilation,
            padding=(window // 2) * dilation,
        )
        tokens = patches.shape[-1]
        patches = patches.view(
            batch,
            self.cross_attention_heads,
            self.cross_attention_head_dim,
            window * window,
            tokens,
        )
        return patches.permute(0, 1, 4, 3, 2).contiguous()

    @staticmethod
    def local_key_offsets(window, dilation, device, dtype):
        radius = window // 2
        offsets = torch.arange(
            -radius, radius + 1, device=device, dtype=dtype
        ) * dilation
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(window * window, 2)

    @staticmethod
    def masked_token_mean(values, mask):
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / denominator

    @staticmethod
    def masked_scalar_mean(values, mask):
        denominator = mask.sum(dim=-1).clamp_min(1.0)
        return (values * mask).sum(dim=-1) / denominator

    @staticmethod
    def masked_scalar_std(values, mask, mean):
        denominator = mask.sum(dim=-1).clamp_min(1.0)
        variance = ((values - mean.unsqueeze(-1)).square() * mask).sum(dim=-1)
        return (variance / denominator).clamp_min(0.0).sqrt()

    @staticmethod
    def masked_scalar_max(values, mask):
        masked = values.masked_fill(mask <= 0.0, -torch.finfo(values.dtype).max)
        maximum = masked.max(dim=-1).values
        has_valid_value = mask.sum(dim=-1) > 0.0
        return torch.where(has_valid_value, maximum, torch.zeros_like(maximum))

    def scale_match_cross_attention_predict(
            self, xin, tar_boxes, ref_boxes, scale_list, H, W, return_debug=False
    ):
        """Ref-to-target local cross-attention residual over the dot-product baseline."""
        baseline_scores, tar_features, ref_context = self.legacy_dot_product_scores(
            xin, tar_boxes, ref_boxes, scale_list, H, W
        )
        baseline_logits = self.scale_preds(baseline_scores)

        batch = tar_features.shape[0]
        scales = scale_list.numel()
        attn_grid = max(1, min(int(self.cross_attention_grid_size), self.grid_size))
        window = self.cross_attention_window_size
        radius = window // 2

        tar_small = F.adaptive_avg_pool2d(tar_features, (attn_grid, attn_grid))
        tar_context = F.pad(
            tar_small, (radius, radius, radius, radius), mode="replicate"
        ) if radius > 0 else tar_small
        ref_context_small = F.adaptive_avg_pool2d(
            ref_context, (attn_grid + 2 * radius, attn_grid + 2 * radius)
        )
        ref_query_map = ref_context_small[
            :, :, radius:radius + attn_grid, radius:radius + attn_grid
        ]

        pair_geometry = self.box_pair_geometry(ref_boxes, tar_boxes, H, W).type_as(xin)
        geometry_embed = self.geometry_proj(pair_geometry)
        position_embed = self.position_proj(
            self.grid_position_tokens(attn_grid, xin.device, xin.dtype)
        )
        scale_value_embed = self.scale_value_embeddings(scale_list, xin.dtype, xin.device)

        ref_raw_tokens = ref_query_map.flatten(start_dim=-2).transpose(1, 2)
        ref_tokens = self.ref_proj(ref_raw_tokens).view(
            batch, scales, attn_grid * attn_grid, self.attn_dim
        )
        ref_tokens = ref_tokens + self.token_type_embed.weight[0].view(1, 1, 1, -1)
        ref_tokens = self.token_norm(
            ref_tokens
            + position_embed[:, None]
            + scale_value_embed.view(1, scales, 1, -1)
            + geometry_embed[:, None, None, :]
        )
        ref_queries = ref_tokens.reshape(
            batch * scales, attn_grid * attn_grid, self.attn_dim
        )

        tar_raw_tokens = tar_context.flatten(start_dim=-2).transpose(1, 2)
        tar_tokens = self.tar_proj(tar_raw_tokens)
        tar_tokens = tar_tokens + self.token_type_embed.weight[1].view(1, 1, -1)
        tar_tokens = self.token_norm(tar_tokens + geometry_embed.unsqueeze(1))

        q = self.split_attention_heads(self.q_proj(self.query_norm(ref_queries)))
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)

        context_grid = attn_grid + 2 * radius
        k_map = self.k_proj(tar_tokens).transpose(1, 2).reshape(
            batch, self.attn_dim, context_grid, context_grid
        )
        v_map = self.v_proj(tar_tokens).transpose(1, 2).reshape(
            batch, self.attn_dim, context_grid, context_grid
        )
        k = self.unfold_local_heads(k_map, window)
        v = self.unfold_local_heads(v_map, window)
        k = k[:, None].expand(-1, scales, -1, -1, -1, -1).reshape(
            batch * scales,
            self.cross_attention_heads,
            attn_grid * attn_grid,
            window * window,
            self.cross_attention_head_dim,
        )
        v = v[:, None].expand(-1, scales, -1, -1, -1, -1).reshape_as(k)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        attn_logits = (q.unsqueeze(-2) * k).sum(dim=-1)
        attn_logits = attn_logits * self.local_attn_logit_scale.exp().clamp(1.0, 20.0)
        attn_logits = attn_logits + self.local_position_bias(
            window, xin.device, xin.dtype
        )
        attn = F.softmax(attn_logits.float(), dim=-1).to(dtype=attn_logits.dtype)
        attn = F.dropout(
            attn, p=self.cross_attention_dropout, training=self.training
        )
        context = (attn.unsqueeze(-1) * v).sum(dim=-2)
        context = self.out_proj(self.merge_attention_heads(context))

        match_tokens = self.context_norm(ref_queries + context)
        aligned_similarity = F.cosine_similarity(
            ref_queries, context, dim=-1, eps=1e-6
        )
        aligned_mean = aligned_similarity.mean(dim=-1).view(batch, scales)
        topk_ratio = max(float(self.similarity_topk_ratio), 0.05)
        topk_count = max(1, int(aligned_similarity.shape[-1] * topk_ratio))
        topk_count = min(topk_count, aligned_similarity.shape[-1])
        aligned_topk = aligned_similarity.topk(
            topk_count, dim=-1
        ).values.mean(dim=-1).view(batch, scales)
        match_delta = (match_tokens - ref_queries).abs().mean(
            dim=(-1, -2)
        ).view(batch, scales)
        attention_confidence = attn.max(dim=-1).values.mean(
            dim=(1, 2)
        ).view(batch, scales)

        geometry_prior, geometry_offset = self.scale_geometry_prior(
            scale_list, pair_geometry
        )
        scale_norm = self.normalized_scale_values(scale_list).to(
            dtype=xin.dtype, device=xin.device
        )
        scale_norm = scale_norm.view(1, scales).expand(batch, -1)
        match_features = torch.stack([
            baseline_scores,
            aligned_mean,
            aligned_topk,
            1.0 - match_delta,
            attention_confidence,
            geometry_prior / 10.0,
            geometry_offset.clamp(-5.0, 5.0) / 5.0,
            scale_norm,
        ], dim=-1)
        match_features = self.match_feature_norm(match_features)
        attention_residual = self.match_scale_mixer(
            match_features.transpose(1, 2)
        ).squeeze(1)

        residual_gate = torch.tanh(self.attention_residual_gate)
        geometry_weight = 2.0 * torch.tanh(self.geometry_prior_weight)
        predictions = (
            baseline_logits
            + residual_gate * attention_residual
            + geometry_weight * geometry_prior
        )
        if not return_debug:
            return predictions

        attn_heads = attn.view(
            batch, scales, self.cross_attention_heads,
            attn_grid * attn_grid, window * window
        )
        attn_summary = attn_heads.mean(dim=(2, 3))
        attn_entropy = -(attn_heads * (attn_heads + 1e-12).log()).sum(dim=-1)
        attn_entropy = attn_entropy.mean(dim=(1, 2, 3))
        debug = {
            "attn": attn_summary,
            "attn_heads": attn_heads,
            "attn_entropy": attn_entropy,
            "baseline_scores": baseline_scores,
            "baseline_logits": baseline_logits,
            "attention_residual": attention_residual,
            "geometry_prior": geometry_prior,
            "geometry_offset": geometry_offset,
            "pair_geometry": pair_geometry,
            "attn_grid": attn_grid,
            "attention_window": window,
            "attention_direction": "ref_to_target",
        }
        return predictions, debug

    def legacy_dot_product_scores(self, xin, tar_boxes, ref_boxes, scale_list, H, W):
        """Return original TSTTC scale scores and reusable ROI features."""
        grid = self.grid_size
        scales = scale_list.numel()
        batch = tar_boxes.shape[0]
        ref_roi_boxes, tar_roi_boxes = self.boxes_sample(ref_boxes, tar_boxes, scale_list, H, W)
        ref_roi_boxes = ref_roi_boxes.type_as(xin)
        tar_roi_boxes = tar_roi_boxes.type_as(xin)
        ref_maps = xin[::2]
        tar_maps = xin[1::2]
        tar_features = roi_align(tar_maps, tar_roi_boxes, (grid, grid))
        tar_match = tar_features.view(batch, 1, 1, self.in_channel, grid, grid)

        if self.shift:
            context_grid = grid + self.shift_kernel_size - 1
            ref_context = roi_align(ref_maps, ref_roi_boxes, (context_grid, context_grid))
            ref_match = self.shift_split(ref_context).view(
                batch, scales, self.shift_kernel_size ** 2, self.in_channel, grid, grid
            )
        else:
            ref_context = roi_align(ref_maps, ref_roi_boxes, (grid, grid))
            ref_match = ref_context.view(batch, scales, 1, self.in_channel, grid, grid)

        if self.normalize_similarity:
            tar_match = F.normalize(tar_match, p=2, dim=3, eps=1e-6)
            ref_match = F.normalize(ref_match, p=2, dim=3, eps=1e-6)
        similarity = (tar_match * ref_match).sum(dim=3).flatten(start_dim=-2)
        mean_score = similarity.mean(dim=-1)
        topk_weight = max(0.0, min(1.0, self.similarity_topk_weight))
        if topk_weight > 0.0:
            topk_ratio = max(0.0, min(1.0, self.similarity_topk_ratio))
            topk_count = max(1, int(similarity.shape[-1] * topk_ratio))
            topk_score = similarity.topk(topk_count, dim=-1).values.mean(dim=-1)
            scores = (1.0 - topk_weight) * mean_score + topk_weight * topk_score
        else:
            scores = mean_score
        scores = scores.max(dim=2).values
        return scores, tar_features, ref_context

    def unfold_local_heads(self, feature_map, window):
        batch_scales, _, height, width = feature_map.shape
        patches = F.unfold(feature_map, kernel_size=window)
        tokens = patches.shape[-1]
        patches = patches.view(
            batch_scales, self.cross_attention_heads, self.cross_attention_head_dim,
            window * window, tokens
        )
        return patches.permute(0, 1, 4, 3, 2).contiguous()

    def local_position_bias(self, window, device, dtype):
        radius = window // 2
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        sigma = max(float(window) * self.cross_attention_position_sigma, 0.5)
        bias = -(xx.square() + yy.square()) / (2.0 * sigma * sigma)
        return bias.reshape(1, 1, 1, window * window)

    def normalized_scale_values(self, scale_list):
        denom = max(float(self.max_scale - self.min_scale), 1e-6)
        return (scale_list - self.min_scale) / denom * 2.0 - 1.0

    def scale_value_embeddings(self, scale_list, dtype, device):
        values = self.normalized_scale_values(scale_list).to(device=device, dtype=dtype)
        return self.scale_value_proj(values.view(-1, 1))

    def scale_geometry_prior(self, scale_list, pair_geometry):
        log_scales = scale_list.to(
            device=pair_geometry.device, dtype=pair_geometry.dtype
        ).clamp_min(1e-6).log()
        # pair_geometry stores log(tar/ref); a candidate scale is ref/tar.
        observed_log_scale = -0.5 * (pair_geometry[:, 0] + pair_geometry[:, 1])
        offset = (
            log_scales.view(1, -1) - observed_log_scale.view(-1, 1)
        ) / self.cross_attention_geometry_sigma
        return -0.5 * offset.square().clamp(max=20.0), offset

    def ref_target_cross_attention_predict(self, xin, tar_boxes, ref_boxes, H, W):
        attn_grid = max(1, min(int(self.cross_attention_grid_size), self.grid_size))
        context_scale = max(float(self.max_scale), 1.0)
        if self.shift:
            context_scale *= (self.grid_size + self.shift_kernel_size - 1) / self.grid_size

        ref_maps = xin[::2, ]
        tar_maps = xin[1::2, ]
        ref_tokens = self.sample_box_tokens(ref_maps, ref_boxes, attn_grid, H, W, context_scale)
        tar_tokens = self.sample_box_tokens(tar_maps, tar_boxes, attn_grid, H, W, 1.0)

        position_embed = self.position_proj(
            self.grid_position_tokens(attn_grid, xin.device, xin.dtype)
        )
        ref_tokens = self.ref_proj(ref_tokens) + self.token_type_embed.weight[0].view(1, 1, -1) + position_embed
        tar_tokens = self.tar_proj(tar_tokens) + self.token_type_embed.weight[1].view(1, 1, -1) + position_embed
        pair_geometry = self.box_pair_geometry(ref_boxes, tar_boxes, H, W).type_as(xin)
        geometry_embed = self.geometry_proj(pair_geometry)
        ref_tokens = self.token_norm(ref_tokens + geometry_embed.unsqueeze(1))
        tar_tokens = self.token_norm(tar_tokens + geometry_embed.unsqueeze(1))

        q = self.q_proj(self.query_norm(ref_tokens))
        k = self.k_proj(tar_tokens)
        v = self.v_proj(tar_tokens)

        q = self.split_attention_heads(q)
        k = self.split_attention_heads(k)
        v = self.split_attention_heads(v)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(self.cross_attention_head_dim, 1))
        attn_logits = attn_logits * self.attn_logit_scale.exp().clamp(max=10.0)
        attn = F.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn, v)
        context = self.merge_attention_heads(context)
        match_tokens = self.context_norm(ref_tokens + self.out_proj(context))

        match_mean = match_tokens.mean(dim=1)
        match_max = match_tokens.max(dim=1).values
        ref_mean = ref_tokens.mean(dim=1)
        tar_mean = tar_tokens.mean(dim=1)
        delta_mean = (match_tokens - ref_tokens).abs().mean(dim=1)
        pair_feature = torch.cat([match_mean, match_max, ref_mean, tar_mean, delta_mean, geometry_embed], dim=-1)
        raw_scale = self.scale_regression_head(pair_feature).squeeze(-1)
        return torch.sigmoid(raw_scale) * (self.max_scale - self.min_scale) + self.min_scale

    def cross_attention_predict(self, xin, tar_boxes, ref_boxes, scale_list, H, W):
        attn_grid = max(1, min(int(self.cross_attention_grid_size), self.grid_size))
        context_scale = max(float(self.max_scale), 1.0)
        if self.shift:
            context_scale *= (self.grid_size + self.shift_kernel_size - 1) / self.grid_size

        ref_maps = xin[::2, ]
        tar_maps = xin[1::2, ]
        ref_tokens = self.sample_box_tokens(ref_maps, ref_boxes, attn_grid, H, W, context_scale)
        tar_tokens = self.sample_box_tokens(tar_maps, tar_boxes, attn_grid, H, W, 1.0)

        position_embed = self.position_proj(
            self.grid_position_tokens(attn_grid, xin.device, xin.dtype)
        )
        ref_tokens = self.ref_proj(ref_tokens) + self.token_type_embed.weight[0].view(1, 1, -1) + position_embed
        tar_tokens = self.tar_proj(tar_tokens) + self.token_type_embed.weight[1].view(1, 1, -1) + position_embed
        pair_geometry = self.box_pair_geometry(ref_boxes, tar_boxes, H, W).type_as(xin)
        geometry_embed = self.geometry_proj(pair_geometry)
        pair_tokens = torch.cat([ref_tokens, tar_tokens], dim=1)
        pair_tokens = self.token_norm(pair_tokens + geometry_embed.unsqueeze(1))

        scale_queries = self.build_scale_queries(scale_list, geometry_embed)
        q = self.q_proj(self.query_norm(scale_queries))
        k = self.k_proj(pair_tokens)
        v = self.v_proj(pair_tokens)

        q = self.split_attention_heads(q)
        k = self.split_attention_heads(k)
        v = self.split_attention_heads(v)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(self.cross_attention_head_dim, 1))
        attn_logits = attn_logits * self.attn_logit_scale.exp().clamp(max=10.0)
        attn = F.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn, v)
        context = self.merge_attention_heads(context)
        scale_states = self.context_norm(scale_queries + self.out_proj(context))
        return self.pred_head(scale_states).squeeze(-1)

    def split_attention_heads(self, x):
        batch, tokens, channels = x.shape
        x = x.view(batch, tokens, self.cross_attention_heads, self.cross_attention_head_dim)
        return x.transpose(1, 2)

    def merge_attention_heads(self, x):
        batch, heads, tokens, channels = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, tokens, heads * channels)

    def grid_position_tokens(self, output_size, device, dtype):
        coords = torch.linspace(-1.0, 1.0, output_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        positions = torch.stack([xx, yy], dim=-1).view(1, output_size * output_size, 2)
        sigma = max(float(self.cross_attention_position_sigma), 1e-3)
        return positions / sigma

    def build_scale_queries(self, scale_list, geometry_embed):
        scale_list = scale_list.to(device=geometry_embed.device, dtype=geometry_embed.dtype)
        denom = max(float(self.max_scale - self.min_scale), 1e-6)
        scale_value = ((scale_list - self.min_scale) / denom * 2.0 - 1.0).view(-1, 1)
        scale_value_embed = self.scale_value_proj(scale_value)

        index_embed = self.scale_index_embed.weight
        if scale_list.shape[0] != index_embed.shape[0]:
            index_embed = F.interpolate(
                index_embed.t().unsqueeze(0),
                size=scale_list.shape[0],
                mode="linear",
                align_corners=True,
            ).squeeze(0).t()
        queries = index_embed.to(dtype=geometry_embed.dtype) + scale_value_embed
        return queries.unsqueeze(0).expand(geometry_embed.shape[0], -1, -1) + geometry_embed.unsqueeze(1)

    def sample_box_tokens(self, feature_maps, boxes, output_size, H, W, context_scale):
        boxes = boxes.to(device=feature_maps.device, dtype=feature_maps.dtype)
        box_xyxy = self.boxes_to_feature_xyxy(boxes, H, W)
        batch_index = boxes[:, 0].long().clamp(min=0, max=feature_maps.shape[0] - 1)
        selected_features = feature_maps.index_select(0, batch_index)

        x1, y1, x2, y2 = box_xyxy.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1).clamp(min=1e-4) * context_scale
        bh = (y2 - y1).clamp(min=1e-4) * context_scale
        x1 = cx - bw * 0.5
        y1 = cy - bh * 0.5
        x2 = cx + bw * 0.5
        y2 = cy + bh * 0.5

        steps = torch.linspace(
            0.5 / output_size,
            1.0 - 0.5 / output_size,
            output_size,
            device=feature_maps.device,
            dtype=feature_maps.dtype,
        )
        yy, xx = torch.meshgrid(steps, steps, indexing="ij")
        sample_x = x1.view(-1, 1, 1) + xx.view(1, output_size, output_size) * (x2 - x1).view(-1, 1, 1)
        sample_y = y1.view(-1, 1, 1) + yy.view(1, output_size, output_size) * (y2 - y1).view(-1, 1, 1)
        grid_x = (sample_x + 0.5) * 2.0 / max(W, 1) - 1.0
        grid_y = (sample_y + 0.5) * 2.0 / max(H, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        sampled = F.grid_sample(
            selected_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.flatten(start_dim=-2).transpose(1, 2)

    def boxes_to_feature_xyxy(self, boxes, H, W):
        box_xyxy = boxes[:, 1:].clone()
        if self.normed_box:
            box_xyxy[:, ::2] = box_xyxy[:, ::2] * W
            box_xyxy[:, 1::2] = box_xyxy[:, 1::2] * H
        return box_xyxy

    def box_pair_geometry(self, ref_boxes, tar_boxes, H, W):
        ref = self.boxes_to_feature_xyxy(ref_boxes, H, W)
        tar = self.boxes_to_feature_xyxy(tar_boxes, H, W)
        eps = ref.new_tensor(1e-6)
        ref_w = (ref[:, 2] - ref[:, 0]).clamp(min=1e-4)
        ref_h = (ref[:, 3] - ref[:, 1]).clamp(min=1e-4)
        tar_w = (tar[:, 2] - tar[:, 0]).clamp(min=1e-4)
        tar_h = (tar[:, 3] - tar[:, 1]).clamp(min=1e-4)
        ref_cx = (ref[:, 0] + ref[:, 2]) * 0.5
        ref_cy = (ref[:, 1] + ref[:, 3]) * 0.5
        tar_cx = (tar[:, 0] + tar[:, 2]) * 0.5
        tar_cy = (tar[:, 1] + tar[:, 3]) * 0.5
        ref_area = ref_w * ref_h
        tar_area = tar_w * tar_h

        inter_x1 = torch.maximum(ref[:, 0], tar[:, 0])
        inter_y1 = torch.maximum(ref[:, 1], tar[:, 1])
        inter_x2 = torch.minimum(ref[:, 2], tar[:, 2])
        inter_y2 = torch.minimum(ref[:, 3], tar[:, 3])
        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h
        iou = inter_area / (ref_area + tar_area - inter_area + eps)

        return torch.stack([
            torch.log((tar_w + eps) / (ref_w + eps)),
            torch.log((tar_h + eps) / (ref_h + eps)),
            torch.log((tar_area + eps) / (ref_area + eps)),
            (tar_cx - ref_cx) / (ref_w + eps),
            (tar_cy - ref_cy) / (ref_h + eps),
            ref_area / max(float(H * W), 1.0),
            tar_area / max(float(H * W), 1.0),
            iou,
        ], dim=-1)

    def get_loss(self, predictions, gts):
        gts = gts.type_as(predictions)
        # print(predictions)
        sample_pairs = gts.shape[0] / self.scale_number
        scale_loss = (
                         self.loss(predictions, gts)
                     ).sum() / sample_pairs

        outputs = predictions.reshape(-1, self.scale_number)
        gts = gts.reshape(-1, self.scale_number)
        _, gt_bin = torch.max(gts, -1, keepdim=False)
        return scale_loss

    def prepare_targets(self, gts, gap, ref_tensor=None):
        if ref_tensor is not None:
            gts = torch.as_tensor(gts, device=ref_tensor.device, dtype=ref_tensor.dtype)
        elif not torch.is_tensor(gts):
            gts = torch.as_tensor(gts)

        if isinstance(gap, (list, tuple)):
            gap_tensor = torch.as_tensor(gap, device=gts.device, dtype=gts.dtype)
        elif torch.is_tensor(gap):
            gap_tensor = gap.to(device=gts.device, dtype=gts.dtype)
        else:
            gap_tensor = torch.full_like(gts, float(gap))
        fps = 10.0 / gap_tensor
        return ttc_to_scale_ratio(gts, fps=fps)

    def get_distribution_loss(self, logits, gt_scales, scale_list):
        step = (self.max_scale - self.min_scale) / (self.scale_number - 1)
        float_idx = ((gt_scales - self.min_scale) / step).clamp(0, self.scale_number - 1)

        idx_l = float_idx.floor().long().clamp(0, self.scale_number - 1)
        idx_r = float_idx.ceil().long().clamp(0, self.scale_number - 1)
        weight_r = float_idx - idx_l
        weight_l = 1.0 - weight_r

        log_probs = F.log_softmax(logits, dim=-1)
        loss_l = F.nll_loss(log_probs, idx_l, reduction="none")
        loss_r = F.nll_loss(log_probs, idx_r, reduction="none")
        return (loss_l * weight_l + loss_r * weight_r).mean()

    def get_scale_match_loss(
            self, logits, gt_scales, scale_list, gt_ttcs=None, frame_gap=None
    ):
        distribution_loss = self.get_distribution_loss(logits, gt_scales, scale_list)
        probabilities = F.softmax(logits, dim=-1)
        scale_list = scale_list.to(device=logits.device, dtype=logits.dtype)
        expected_scale = (probabilities * scale_list.view(1, -1)).sum(dim=-1)

        total_loss = distribution_loss
        if self.cross_attention_reg_loss_weight > 0.0:
            step = max(
                (self.max_scale - self.min_scale) / max(self.scale_number - 1, 1),
                1e-6,
            )
            normalized_error = (
                expected_scale - gt_scales.type_as(expected_scale)
            ) / step
            regression_loss = F.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                beta=1.0,
                reduction="mean",
            )
            total_loss = (
                total_loss
                + self.cross_attention_reg_loss_weight * regression_loss
            )

        if (
                self.cross_attention_ttc_loss_weight > 0.0
                and gt_ttcs is not None
                and frame_gap is not None
        ):
            ttc_loss = self.get_ttc_metric_loss(
                expected_scale, gt_ttcs, frame_gap
            )
            total_loss = (
                total_loss
                + self.cross_attention_ttc_loss_weight * ttc_loss
            )
        return total_loss

    def get_fps_tensor(self, frame_gap, reference):
        if isinstance(frame_gap, (list, tuple)):
            gap = torch.as_tensor(
                frame_gap, dtype=reference.dtype, device=reference.device
            ).view(-1)
        elif torch.is_tensor(frame_gap):
            gap = frame_gap.to(
                device=reference.device, dtype=reference.dtype
            ).view(-1)
        else:
            gap = torch.full_like(reference.view(-1), float(frame_gap))
        if gap.numel() == 1 and reference.numel() != 1:
            gap = gap.expand(reference.numel())
        return 10.0 / gap.clamp_min(1.0)

    def get_ttc_metric_sample_weights(self, gt_ttcs, device, dtype):
        gt_abs = torch.as_tensor(
            gt_ttcs, device=device, dtype=dtype
        ).view(-1).abs()
        weights = torch.full_like(gt_abs, self.ttc_metric_short_loss_weight)
        weights = torch.where(
            gt_abs >= self.ttc_metric_mid_ttc_abs_thresh,
            torch.full_like(weights, self.ttc_metric_mid_loss_weight),
            weights,
        )
        weights = torch.where(
            gt_abs >= self.ttc_metric_long_ttc_abs_thresh,
            torch.full_like(weights, self.ttc_metric_long_loss_weight),
            weights,
        )
        if self.ttc_metric_tail_ttc_abs_thresh > 0:
            weights = torch.where(
                gt_abs >= self.ttc_metric_tail_ttc_abs_thresh,
                torch.full_like(weights, self.ttc_metric_tail_loss_weight),
                weights,
            )
        return weights

    def get_ttc_metric_loss(self, pred_scales, gt_ttcs, frame_gap):
        pred_scales = pred_scales.view(-1).clamp_min(1e-6)
        gt_ttcs = torch.as_tensor(
            gt_ttcs, device=pred_scales.device, dtype=pred_scales.dtype
        ).view(-1)
        fps = self.get_fps_tensor(frame_gap, pred_scales)
        pred_ttcs = scale_ratio_to_ttc(pred_scales, fps=fps)
        pred_ttcs = pred_ttcs.clamp(-self.ttc_metric_clip, self.ttc_metric_clip)
        gt_ttcs = gt_ttcs.clamp(-self.ttc_metric_clip, self.ttc_metric_clip)
        denominator = gt_ttcs.abs().clamp_min(self.ttc_metric_min_denom)
        relative_delta = (pred_ttcs - gt_ttcs) / denominator
        abs_delta = relative_delta.abs()
        beta = self.ttc_metric_huber_beta
        loss = torch.where(
            abs_delta < beta,
            0.5 * abs_delta.square() / beta,
            abs_delta - 0.5 * beta,
        )
        weights = self.get_ttc_metric_sample_weights(
            gt_ttcs, pred_scales.device, pred_scales.dtype
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-12)

    def shift_split(self, ref_features):
        """Split a context ROI into the same local shifts used by the baseline."""
        windows = []
        for row in range(self.shift_kernel_size):
            for col in range(self.shift_kernel_size):
                windows.append(
                    ref_features[:, :, row:row + self.grid_size, col:col + self.grid_size]
                )
        return torch.stack(windows, dim=1)

    def gt_to_one_hot(self, gts, gap, scale_list, ref_tensor=None):
        if ref_tensor is not None:
            gts = torch.as_tensor(gts, device=ref_tensor.device, dtype=ref_tensor.dtype)
            scale_list = scale_list.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        elif not torch.is_tensor(gts):
            gts = torch.as_tensor(gts)

        if isinstance(gap, (list, tuple)):
            gap_tensor = torch.as_tensor(gap, device=gts.device, dtype=gts.dtype)
        elif torch.is_tensor(gap):
            gap_tensor = gap.to(device=gts.device, dtype=gts.dtype)
        else:
            gap_tensor = torch.full_like(gts, float(gap))

        scale_number, gt_number = self.scale_number, gts.numel()
        scale_gt = ttc_to_scale_ratio(gts, fps=10.0 / gap_tensor)
        range_tensor = scale_list.detach().clone().view(1, scale_number).repeat(gt_number, 1)
        gts_tensor = scale_gt.view(gt_number, 1).repeat(1, scale_number).type_as(range_tensor)

        ones_mat = torch.ones_like(gts_tensor)
        zero_mat = torch.zeros_like(ones_mat)
        dist_mat = torch.abs(range_tensor - gts_tensor)
        min_bin = self.smoother_factor*(self.max_scale - self.min_scale) / (self.scale_number - 1)
        dist_tensor = torch.where(dist_mat > min_bin, zero_mat, min_bin - dist_mat) * (1 / min_bin)
        gt_one_hot = dist_tensor.view(-1, 1)
        return gt_one_hot

    def get_scale_list(self, S):
        if S == self.scale_number:
            return self._default_scale_list
        return torch.linspace(
            self.min_scale,
            self.max_scale,
            S,
            device=self._default_scale_list.device,
            dtype=self._default_scale_list.dtype,
        )

    def boxes_sample(self,ref_boxes,tar_boxes,scale_list,H=576,W=1024):
        '''
        convert the boxes to multi-scale format
        :param tar_boxes: [K,5]
        :param ref_boxes: [K,5]
        :return: [K,scale_number,5]
        '''
        def to_algin_space(center_boxes,hw_boxes,scale_list):
            box_num = center_boxes.shape[0]
            box_xx = ref_boxes.new_tensor([-0.5 * W, 0.5 * W])
            box_yy = ref_boxes.new_tensor([-0.5 * H, 0.5 * H])
            scale_count = scale_list.shape[-1]
            box_xx = box_xx.view(1, 1, 2).expand(box_num, scale_count, 2)
            box_yy = box_yy.view(1, 1, 2).expand(box_num, scale_count, 2)  # [box_num,scale_num,2]
            if len(scale_list.shape) == 1:
                scaled_w = scale_list.view(1, scale_count, 1).expand(box_num, scale_count, 2).type_as(ref_boxes)
                scale_h = scale_list.view(1, scale_count, 1).expand(box_num, scale_count, 2).type_as(ref_boxes)
            else:#dynamic mode
                scaled_w = scale_list.unsqueeze(-1).expand(-1, -1, 2).type_as(ref_boxes)
                scale_h = scale_list.unsqueeze(-1).expand(-1, -1, 2).type_as(ref_boxes)  # [box_num,scale_num,2]
            box_xx, box_yy = torch.mul(scaled_w, box_xx), torch.mul(scale_h, box_yy)

            tar_boxes_wh = hw_boxes[:, 2:] - hw_boxes[:, :2]  # box_num,2
            if self.shift and scale_list.shape[-1] > 1:
                tar_boxes_wh = tar_boxes_wh*(self.grid_size + self.shift_kernel_size - 1) /self.grid_size#
            tar_boxes_wh = tar_boxes_wh.unsqueeze(1).expand(-1, scale_list.shape[-1], -1)  # box_num,scale_num,2
            box_xx = box_xx * tar_boxes_wh[:, :, 0].unsqueeze(-1) / W  # box_num,scale_num,2
            box_yy = box_yy * tar_boxes_wh[:, :, 1].unsqueeze(-1) / H  # box_num,scale_num,2
            box_cx, box_cy = (center_boxes[:, 2] + center_boxes[:, 0]) / 2, (center_boxes[:, 3] + center_boxes[:, 1]) / 2
            box_cx = box_cx.unsqueeze(1).unsqueeze(-1).expand(-1, scale_list.shape[-1], 2)
            box_cy = box_cy.unsqueeze(1).unsqueeze(-1).expand(-1, scale_list.shape[-1], 2)
            box_xx, box_yy = box_xx + box_cx, box_yy + box_cy  # box_num,scale_num,2
            align_boxes = torch.stack([box_xx[:, :, 0], box_yy[:, :, 0], box_xx[:, :, 1], box_yy[:, :, 1]],dim=-1)  # box_num,scale_num,4
            return align_boxes

        batch_index = tar_boxes[:,:1]
        tar_boxes = tar_boxes[:,1:].clone()
        ref_boxes = ref_boxes[:,1:].clone()
        if self.normed_box:
            tar_boxes[:,::2] = tar_boxes[:,::2]*W
            tar_boxes[:,1::2] = tar_boxes[:,1::2]*H
            ref_boxes[:,::2] = ref_boxes[:,::2]*W
            ref_boxes[:,1::2] = ref_boxes[:,1::2]*H
        scaled_ref_boxes = to_algin_space(ref_boxes,tar_boxes,scale_list).view(-1,4)
        scaled_tar_boxes = to_algin_space(tar_boxes,tar_boxes,ref_boxes.new_tensor([1.0])).view(-1,4)
        scale_count = scale_list.shape[-1]
        ref_align_boxes = torch.cat([batch_index.repeat([1,scale_count]).view([-1,1]),scaled_ref_boxes],dim=-1)
        tar_align_boxes = torch.cat([batch_index.repeat([1,1]),scaled_tar_boxes],dim=-1)
        return ref_align_boxes,tar_align_boxes
