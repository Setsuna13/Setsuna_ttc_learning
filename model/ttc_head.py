from torch import nn
from .network_blocks import FocalLoss
from torchvision.ops import roi_align
import torch
import torch.nn.functional as F
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
            use_per_bin_residual_head = False,
            residual_bin_num = 31,
            residual_scale_range = 0.03,
            residual_loss_weight = 0.3,
            final_scale_loss_weight = 0.5,
            residual_short_loss_weight = 0.05,
            residual_mid_ttc_abs_thresh = 3.0,
            residual_mid_loss_weight = 0.3,
            residual_long_ttc_abs_thresh = 6.0,
            residual_long_loss_weight = 1.0,
            residual_tail_ttc_abs_thresh = 12.0,
            residual_tail_loss_weight = 1.2,
            scale_bin_mode = "linear",
            scale_bin_density_power = 2.5,
            scale_bin_center = 1.0,
            use_ttc_metric_loss = False,
            ttc_metric_loss_weight = 1.0,
            ttc_metric_clip = 20.0,
            ttc_metric_min_denom = 1.0,
            ttc_metric_huber_beta = 0.1,
            ttc_metric_short_loss_weight = 0.02,
            ttc_metric_mid_ttc_abs_thresh = 3.0,
            ttc_metric_mid_loss_weight = 0.25,
            ttc_metric_long_ttc_abs_thresh = 6.0,
            ttc_metric_long_loss_weight = 1.0,
            ttc_metric_tail_ttc_abs_thresh = 12.0,
            ttc_metric_tail_loss_weight = 1.2,
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
        self.scale_preds = nn.Linear(scale_number, scale_number)
        self.smoother_factor = smoother_factor # training only
        self.head_type = head_type
        self.normalize_similarity = normalize_similarity
        self.similarity_topk_ratio = similarity_topk_ratio
        self.similarity_topk_weight = similarity_topk_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.use_per_bin_residual_head = bool(use_per_bin_residual_head)
        self.residual_bin_num = max(3, int(residual_bin_num))
        self.residual_scale_range = max(float(residual_scale_range), 1e-6)
        self.residual_loss_weight = float(residual_loss_weight)
        self.final_scale_loss_weight = float(final_scale_loss_weight)
        self.residual_short_loss_weight = float(residual_short_loss_weight)
        self.residual_mid_ttc_abs_thresh = float(residual_mid_ttc_abs_thresh)
        self.residual_mid_loss_weight = float(residual_mid_loss_weight)
        self.residual_long_ttc_abs_thresh = float(residual_long_ttc_abs_thresh)
        self.residual_long_loss_weight = float(residual_long_loss_weight)
        self.residual_tail_ttc_abs_thresh = float(residual_tail_ttc_abs_thresh)
        self.residual_tail_loss_weight = float(residual_tail_loss_weight)
        self.scale_bin_mode = str(scale_bin_mode)
        self.scale_bin_density_power = max(float(scale_bin_density_power), 1.0)
        self.scale_bin_center = float(scale_bin_center)
        self.use_ttc_metric_loss = bool(use_ttc_metric_loss)
        self.ttc_metric_loss_weight = float(ttc_metric_loss_weight)
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
        if self.use_per_bin_residual_head:
            self.residual_preds = nn.Linear(scale_number, scale_number * self.residual_bin_num)
            nn.init.zeros_(self.residual_preds.weight)
            nn.init.zeros_(self.residual_preds.bias)

    def forward(self, xin, tar_boxes,ref_boxes,ttc_imu =None,**kwargs):
        '''

        :param xin: input features from backbone
        :param tar_boxes: input boxes in roi_align format #[K,5]
        :param ref_boxes:
        :return: predicted scale confidences
        '''

        C,H,W = xin.shape[-3:]
        G,S,Box = self.grid_size,self.scale_number, tar_boxes.shape[0]
        scale_list = self.get_scale_list(S)
        ref_boxes,tar_boxes = self.boxes_sample(ref_boxes,tar_boxes,scale_list,H,W)
        ref_boxes,tar_boxes = ref_boxes.type_as(xin),tar_boxes.type_as(xin)
        tar_features = roi_align(xin[1::2, ], tar_boxes, (G, G))
        tar_tensor = tar_features.view([-1, 1, C, G, G])  # boxes,1,C,H,W
        tar_tensor = torch.flatten(tar_tensor, start_dim=3).permute([0,1,3,2]).unsqueeze(3)#boxes,1,H*W,1,C
        if self.shift:
            ref_features = roi_align(xin[::2, ], ref_boxes, (G+self.shift_kernel_size-1, G+self.shift_kernel_size-1))
            ref_features = self.shift_split(ref_features).view([Box,S,-1,C,G,G])
            ref_features = ref_features.reshape([Box,-1,C,G,G])
            ref_features = ref_features.flatten(start_dim=-2).permute([0,1,3,2]).unsqueeze(-1) #boxes,scale_number*shift_kernel_size**2,H*W,C,1
        else:
            ref_features = roi_align(xin[::2, ], ref_boxes, (G, G)) #boxes*scale_number,C,H,W
            ref_features = ref_features.view([Box,-1,xin.shape[1], G, G])#boxes,scale_number,C,H,W
            ref_features = ref_features.flatten(start_dim=-2).permute([0,1,3,2]).unsqueeze(-1)#boxes,scale_number,H*W,C,1
        if self.normalize_similarity:
            tar_tensor = F.normalize(tar_tensor, p=2, dim=-1, eps=1e-6)
            ref_features = F.normalize(ref_features, p=2, dim=-2, eps=1e-6)
        #TODO add codes for other distance type here
        simlarities_map = torch.matmul(tar_tensor, ref_features).squeeze(-1).squeeze(-1) #boxes,scale_number,H*W

        mean_score = torch.mean(simlarities_map,dim=-1) #boxes,scale_number
        topk_weight = max(0.0, min(1.0, self.similarity_topk_weight))
        if topk_weight > 0:
            topk_ratio = max(0.0, min(1.0, self.similarity_topk_ratio))
            topk_count = max(1, int(simlarities_map.shape[-1] * topk_ratio))
            topk_score = simlarities_map.topk(k=topk_count, dim=-1).values.mean(dim=-1)
            simlarities_scale = (1.0 - topk_weight) * mean_score + topk_weight * topk_score
        else:
            simlarities_scale = mean_score
        if self.shift:
            simlarities_scale = torch.max(simlarities_scale.view([-1,self.scale_number, self.shift_kernel_size ** 2]), dim=-1).values

        predictions = self.scale_preds(simlarities_scale)
        residual_logits = None
        if self.use_per_bin_residual_head and self.head_type == "distribution":
            residual_logits = self.residual_preds(simlarities_scale).view(
                -1, self.scale_number, self.residual_bin_num
            )
        if self.training:
            if self.head_type == "distribution":
                dict_annos = kwargs.get('dictAnnos')
                if dict_annos is not None:
                    frame_gap = dict_annos['frame_gap']
                else:
                    frame_gap = self.sequence_len - 1
                scale_gts = None if dict_annos is None else dict_annos.get('scale_gt')
                gt_scales = self.prepare_targets(
                    ttc_imu, frame_gap, scale_gts=scale_gts
                ).type_as(predictions)
                distribution_loss = self.get_distribution_loss(predictions, gt_scales, scale_list)
                total_loss = distribution_loss
                pred_scales = None
                base_probs = None
                if self.use_per_bin_residual_head:
                    pred_scales, base_probs = self.apply_per_bin_residual(
                        predictions, residual_logits, scale_list
                    )
                    sample_weights = self.get_residual_sample_weights(
                        ttc_imu, predictions.device, predictions.dtype
                    )
                    residual_loss = self.get_per_bin_residual_loss(
                        residual_logits, gt_scales, scale_list, base_probs.detach(), sample_weights
                    )
                    final_scale_loss = self.get_weighted_scale_loss(
                        pred_scales, gt_scales, sample_weights
                    )
                    total_loss = (
                        total_loss
                        + self.residual_loss_weight * residual_loss
                        + self.final_scale_loss_weight * final_scale_loss
                    )
                if self.use_ttc_metric_loss:
                    if pred_scales is None:
                        pred_scales, base_probs = self.apply_distribution_prediction(predictions, scale_list)
                    metric_loss = self.get_ttc_metric_loss(pred_scales, ttc_imu, frame_gap)
                    total_loss = total_loss + self.ttc_metric_loss_weight * metric_loss
                return total_loss

            predictions = predictions.view(-1,1)
            dict_annos = kwargs.get('dictAnnos')
            if dict_annos is not None:
                frame_gap = dict_annos['frame_gap']
            else:
                frame_gap = self.sequence_len-1
            scale_gts = None if dict_annos is None else dict_annos.get('scale_gt')
            gt_one_hot = self.gt_to_one_hot(
                ttc_imu, gap=frame_gap, scale_list=scale_list, scale_gts=scale_gts
            )
            scale_loss = self.get_loss(predictions, gt_one_hot)
            return scale_loss
        if self.head_type == "distribution":
            base_probs = F.softmax(predictions, dim=-1)
            if self.use_per_bin_residual_head:
                pred_scales, _ = self.apply_per_bin_residual(predictions, residual_logits, scale_list)
                return base_probs, scale_list, pred_scales
            return base_probs, scale_list, None
        return predictions.sigmoid(), scale_list, None

    def shift_split(self,ref_features):
        '''
        split the reference features into shift_kernel_size*shift_kernel_size parts
        :param ref_features: [N,C,self.grid_size+self.shift_kernel_size-1, self.grid_size+self.shift_kernel_size-1]
        :return: [N,self.shift_kernel_size**2,C,H,W]
        '''
        tmp_list = []
        for i in range(self.shift_kernel_size):
            for j in range(self.shift_kernel_size):
                tmp_list.append(ref_features[:,:,i:i+self.grid_size,j:j+self.grid_size])
        ref_features = torch.stack(tmp_list,dim=1) #N,shift_kernel_size*shift_kernel_size,C,H,W
        return ref_features

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

    def prepare_targets(self, gts, gap, scale_gts=None):
        if scale_gts is not None and len(scale_gts) > 0:
            if len(scale_gts) != len(gts):
                raise ValueError(
                    "scale_gt count {} does not match TTC count {}".format(
                        len(scale_gts), len(gts)
                    )
                )
            return torch.as_tensor(scale_gts, dtype=torch.float32).view(-1)
        if isinstance(gap, (list, tuple)):
            scale_gt = [ttc_to_scale_ratio(ttc, fps=10 / float(tmp_gap)) for ttc, tmp_gap in zip(gts, gap)]
        elif torch.is_tensor(gap):
            scale_gt = [ttc_to_scale_ratio(ttc, fps=10 / float(tmp_gap)) for ttc, tmp_gap in zip(gts, gap)]
        else:
            scale_gt = [ttc_to_scale_ratio(ttc, fps=10 / gap) for ttc in gts]
        return torch.tensor(scale_gt)

    def get_distribution_loss(self, logits, gt_scales, scale_list):
        scale_list = scale_list.to(device=logits.device, dtype=logits.dtype)
        gt_scales = gt_scales.to(device=logits.device, dtype=logits.dtype).view(-1)
        gt_scales = gt_scales.clamp(scale_list[0], scale_list[-1])

        idx_r = torch.searchsorted(scale_list, gt_scales, right=False).clamp(1, self.scale_number - 1)
        idx_l = (idx_r - 1).clamp(0, self.scale_number - 1)
        scale_l = scale_list[idx_l]
        scale_r = scale_list[idx_r]
        weight_r = (gt_scales - scale_l) / (scale_r - scale_l).clamp_min(1e-12)
        weight_r = weight_r.clamp(0.0, 1.0)
        weight_l = 1.0 - weight_r

        log_probs = F.log_softmax(logits, dim=-1)
        loss_l = F.nll_loss(log_probs, idx_l, reduction="none")
        loss_r = F.nll_loss(log_probs, idx_r, reduction="none")
        return (loss_l * weight_l + loss_r * weight_r).mean()

    def apply_distribution_prediction(self, logits, scale_list):
        scale_list = scale_list.to(device=logits.device, dtype=logits.dtype)
        probs = F.softmax(logits, dim=-1)
        pred_scales = torch.sum(probs * scale_list.view(1, -1), dim=-1)
        return pred_scales, probs

    def get_fps_tensor(self, frame_gap, reference):
        if isinstance(frame_gap, (list, tuple)):
            gap = torch.as_tensor(frame_gap, dtype=reference.dtype, device=reference.device).view(-1)
        elif torch.is_tensor(frame_gap):
            gap = frame_gap.to(device=reference.device, dtype=reference.dtype).view(-1)
        else:
            gap = torch.full_like(reference.view(-1), float(frame_gap))
        if gap.numel() == 1 and reference.numel() != 1:
            gap = gap.expand(reference.numel())
        gap = gap.clamp_min(1.0)
        return 10.0 / gap

    def get_ttc_metric_sample_weights(self, gt_ttcs, device, dtype):
        gt_abs = torch.as_tensor(gt_ttcs, device=device, dtype=dtype).view(-1).abs()
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
        gt_ttcs = torch.as_tensor(gt_ttcs, device=pred_scales.device, dtype=pred_scales.dtype).view(-1)
        fps = self.get_fps_tensor(frame_gap, pred_scales)
        pred_ttcs = scale_ratio_to_ttc(pred_scales, fps=fps)
        pred_ttcs = pred_ttcs.clamp(-self.ttc_metric_clip, self.ttc_metric_clip)
        gt_ttcs = gt_ttcs.clamp(-self.ttc_metric_clip, self.ttc_metric_clip)
        denom = gt_ttcs.abs().clamp_min(self.ttc_metric_min_denom)
        rel_delta = (pred_ttcs - gt_ttcs) / denom
        abs_delta = rel_delta.abs()
        beta = self.ttc_metric_huber_beta
        loss = torch.where(
            abs_delta < beta,
            0.5 * abs_delta.pow(2) / beta,
            abs_delta - 0.5 * beta,
        )
        weights = self.get_ttc_metric_sample_weights(gt_ttcs, pred_scales.device, pred_scales.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1e-12)

    def get_residual_list(self, device=None, dtype=None):
        return torch.linspace(
            -self.residual_scale_range,
            self.residual_scale_range,
            self.residual_bin_num,
            device=device,
            dtype=dtype,
        )

    def apply_per_bin_residual(self, base_logits, residual_logits, scale_list):
        scale_list = scale_list.to(device=base_logits.device, dtype=base_logits.dtype)
        residual_list = self.get_residual_list(base_logits.device, base_logits.dtype)
        base_probs = F.softmax(base_logits, dim=-1)
        residual_probs = F.softmax(residual_logits, dim=-1)
        corrected_scales = scale_list.view(1, -1, 1) + residual_list.view(1, 1, -1)
        corrected_scales = corrected_scales.clamp_min(1e-6)
        pred_scales = (base_probs.unsqueeze(-1) * residual_probs * corrected_scales).sum(dim=(1, 2))
        return pred_scales, base_probs

    def get_residual_sample_weights(self, gt_ttcs, device, dtype):
        if gt_ttcs is None:
            return torch.ones(1, device=device, dtype=dtype)
        gt_abs = torch.as_tensor(gt_ttcs, device=device, dtype=dtype).view(-1).abs()
        weights = torch.full_like(gt_abs, self.residual_short_loss_weight)
        weights = torch.where(
            gt_abs >= self.residual_mid_ttc_abs_thresh,
            torch.full_like(weights, self.residual_mid_loss_weight),
            weights,
        )
        weights = torch.where(
            gt_abs >= self.residual_long_ttc_abs_thresh,
            torch.full_like(weights, self.residual_long_loss_weight),
            weights,
        )
        if self.residual_tail_ttc_abs_thresh > 0:
            weights = torch.where(
                gt_abs >= self.residual_tail_ttc_abs_thresh,
                torch.full_like(weights, self.residual_tail_loss_weight),
                weights,
            )
        return weights

    def get_weighted_scale_loss(self, pred_scales, gt_scales, sample_weights):
        pred_scales = pred_scales.view(-1)
        gt_scales = gt_scales.to(device=pred_scales.device, dtype=pred_scales.dtype).view(-1)
        sample_weights = sample_weights.to(device=pred_scales.device, dtype=pred_scales.dtype).view(-1)
        scale_step = max((self.max_scale - self.min_scale) / max(self.scale_number - 1, 1), 1e-6)
        loss = torch.abs(pred_scales - gt_scales) / scale_step
        return (loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)

    def get_per_bin_residual_loss(self, residual_logits, gt_scales, scale_list, bin_weights, sample_weights=None):
        scale_list = scale_list.to(device=residual_logits.device, dtype=residual_logits.dtype)
        gt_scales = gt_scales.to(device=residual_logits.device, dtype=residual_logits.dtype).view(-1, 1)
        bin_weights = bin_weights.to(device=residual_logits.device, dtype=residual_logits.dtype)
        if sample_weights is None:
            sample_weights = torch.ones(gt_scales.shape[0], device=residual_logits.device, dtype=residual_logits.dtype)
        else:
            sample_weights = sample_weights.to(device=residual_logits.device, dtype=residual_logits.dtype).view(-1)

        residual_targets = (gt_scales - scale_list.view(1, -1)).clamp(
            -self.residual_scale_range, self.residual_scale_range
        )
        step = (2.0 * self.residual_scale_range) / (self.residual_bin_num - 1)
        float_idx = ((residual_targets + self.residual_scale_range) / step).clamp(
            0, self.residual_bin_num - 1
        )
        idx_l = float_idx.floor().long().clamp(0, self.residual_bin_num - 1)
        idx_r = float_idx.ceil().long().clamp(0, self.residual_bin_num - 1)
        weight_r = float_idx - idx_l
        weight_l = 1.0 - weight_r

        log_probs = F.log_softmax(residual_logits, dim=-1).reshape(-1, self.residual_bin_num)
        loss_l = F.nll_loss(log_probs, idx_l.reshape(-1), reduction="none").view_as(float_idx)
        loss_r = F.nll_loss(log_probs, idx_r.reshape(-1), reduction="none").view_as(float_idx)
        residual_loss = loss_l * weight_l + loss_r * weight_r
        weights = bin_weights * sample_weights.view(-1, 1)
        return (residual_loss * weights).sum() / weights.sum().clamp_min(1e-12)

    def gt_to_one_hot(self, gts, gap, scale_list, scale_gts=None):
        if scale_gts is not None and len(scale_gts) > 0:
            if len(scale_gts) != len(gts):
                raise ValueError(
                    "scale_gt count {} does not match TTC count {}".format(
                        len(scale_gts), len(gts)
                    )
                )
            scale_gt = scale_gts
        elif type(gap) is list:
            scale_gt = [ttc_to_scale_ratio(ttc, fps=10 / tmp_gap) for ttc,tmp_gap in zip(gts,gap)]
        else:
            scale_gt = [ttc_to_scale_ratio(ttc, fps=10 / gap) for ttc in gts]
        scale_number,gt_number,range_list = self.scale_number,len(gts),scale_list
        if torch.is_tensor(range_list):
            range_tensor = range_list.detach().clone().view([-1, scale_number])
        else:
            range_tensor = torch.tensor(range_list).view([-1, scale_number])
        range_tensor = range_tensor.repeat(gt_number, 1)
        gts_tensor = torch.tensor(scale_gt).view(gt_number, -1).repeat([1, scale_number]).type_as(range_tensor)

        ones_mat = torch.ones_like(gts_tensor)
        zero_mat = torch.zeros_like(ones_mat)
        dist_mat = torch.abs(range_tensor - gts_tensor)
        min_bin = self.smoother_factor*(self.max_scale - self.min_scale) / (self.scale_number - 1)
        dist_tensor = torch.where(dist_mat > min_bin, zero_mat, min_bin - dist_mat) * (1 / min_bin)
        gt_one_hot = dist_tensor.view(-1, 1)
        return gt_one_hot

    def get_scale_list(self, S):
        mode = self.scale_bin_mode.lower()
        if mode in ("linear", "uniform"):
            return torch.linspace(self.min_scale, self.max_scale, S)
        if mode in ("center_dense", "ttc_aware", "ttc_aware_center_dense"):
            return self.get_center_dense_scale_list(S)
        raise ValueError("Unsupported scale_bin_mode: {}".format(self.scale_bin_mode))

    def get_center_dense_scale_list(self, S):
        if S <= 2:
            return torch.linspace(self.min_scale, self.max_scale, S)
        center = min(max(self.scale_bin_center, self.min_scale + 1e-6), self.max_scale - 1e-6)
        left_range = center - self.min_scale
        right_range = self.max_scale - center
        total_range = left_range + right_range
        if left_range <= 0 or right_range <= 0 or total_range <= 0:
            return torch.linspace(self.min_scale, self.max_scale, S)

        left_bins = int(round((S - 1) * left_range / total_range)) + 1
        left_bins = min(max(left_bins, 2), S - 1)
        right_bins = S - left_bins + 1
        power = self.scale_bin_density_power

        left_u = torch.linspace(0.0, 1.0, left_bins)
        right_u = torch.linspace(0.0, 1.0, right_bins)
        left = center - left_range * torch.pow(1.0 - left_u, power)
        right = center + right_range * torch.pow(right_u, power)
        return torch.cat([left, right[1:]], dim=0)

    def boxes_sample(self,ref_boxes,tar_boxes,scale_list,H=576,W=1024):
        '''
        convert the boxes to multi-scale format
        :param tar_boxes: [K,5]
        :param ref_boxes: [K,5]
        :return: [K,scale_number,5]
        '''
        def to_algin_space(center_boxes,hw_boxes,scale_list):
            box_num = center_boxes.shape[0]
            box_xx, box_yy = torch.tensor([-0.5 * W, 0.5 * W]).type_as(ref_boxes), torch.tensor([-0.5 * H, 0.5 * H]).type_as(ref_boxes)
            box_xx = box_xx.unsqueeze(0).unsqueeze(0).repeat(box_num, scale_list.shape[-1],1)
            box_yy = box_yy.unsqueeze(0).unsqueeze(0).repeat(box_num, scale_list.shape[-1],1)  # [box_num,scale_num,2]
            if len(scale_list.shape) == 1:
                scaled_w = scale_list.unsqueeze(1).unsqueeze(0).repeat([box_num, 1, 2]).type_as(ref_boxes)
                scale_h = scale_list.unsqueeze(1).unsqueeze(0).repeat([box_num, 1, 2]).type_as(ref_boxes)  # [box_num,scale_num,2]
            else:#dynamic mode
                scaled_w = scale_list.unsqueeze(-1).repeat([1, 1, 2]).type_as(ref_boxes)
                scale_h = scale_list.unsqueeze(-1).repeat([1, 1, 2]).type_as(ref_boxes)  # [box_num,scale_num,2]
            box_xx, box_yy = torch.mul(scaled_w, box_xx), torch.mul(scale_h, box_yy)

            tar_boxes_wh = hw_boxes[:, 2:] - hw_boxes[:, :2]  # box_num,2
            if self.shift and scale_list.shape[-1] > 1:
                tar_boxes_wh = tar_boxes_wh*(self.grid_size + self.shift_kernel_size - 1) /self.grid_size#
            tar_boxes_wh = tar_boxes_wh.unsqueeze(1).repeat([1, scale_list.shape[-1], 1])  # box_num,scale_num,2
            box_xx = box_xx * tar_boxes_wh[:, :, 0].unsqueeze(-1) / W  # box_num,scale_num,2
            box_yy = box_yy * tar_boxes_wh[:, :, 1].unsqueeze(-1) / H  # box_num,scale_num,2
            box_cx, box_cy = (center_boxes[:, 2] + center_boxes[:, 0]) / 2, (center_boxes[:, 3] + center_boxes[:, 1]) / 2
            box_cx = box_cx.unsqueeze(1).unsqueeze(-1).repeat([1, scale_list.shape[-1], 2])
            box_cy = box_cy.unsqueeze(1).unsqueeze(-1).repeat([1, scale_list.shape[-1], 2])
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
        scaled_tar_boxes = to_algin_space(tar_boxes,tar_boxes,torch.tensor([1.0])).view(-1,4)
        ref_align_boxes = torch.cat([batch_index.repeat([1,self.scale_number]).view([-1,1]),scaled_ref_boxes],dim=-1)
        tar_align_boxes = torch.cat([batch_index.repeat([1,1]),scaled_tar_boxes],dim=-1)
        return ref_align_boxes,tar_align_boxes
