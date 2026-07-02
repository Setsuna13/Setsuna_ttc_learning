from torch import nn
from .network_blocks import FocalLoss
import torch
import torch.nn.functional as F
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
        if not use_cross_attention:
            raise ValueError("The legacy ROI Align enumeration head has been removed; use_cross_attention must be True.")
        self.use_cross_attention = use_cross_attention
        self.cross_attention_grid_size = cross_attention_grid_size
        self.cross_attention_position_sigma = cross_attention_position_sigma
        self.ce_loss = nn.CrossEntropyLoss()
        self.register_buffer(
            "_default_scale_list",
            torch.linspace(self.min_scale, self.max_scale, self.scale_number),
            persistent=False,
        )

        attn_dim = cross_attention_dim or in_channel
        self.attn_dim = attn_dim
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
        self.attn_logit_scale = nn.Parameter(torch.zeros(1))

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
        predictions = self.cross_attention_predict(xin, tar_boxes, ref_boxes, scale_list, H, W)
        if self.training:
            if self.head_type == "distribution":
                if 'dictAnnos' in kwargs:
                    frame_gap = kwargs['dictAnnos']['frame_gap']
                else:
                    frame_gap = self.sequence_len - 1
                gt_scales = self.prepare_targets(ttc_imu, frame_gap, predictions)
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

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(self.attn_dim, 1))
        attn_logits = attn_logits * self.attn_logit_scale.exp().clamp(max=10.0)
        attn = F.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn, v)
        scale_states = self.context_norm(scale_queries + self.out_proj(context))
        return self.pred_head(scale_states).squeeze(-1)

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
