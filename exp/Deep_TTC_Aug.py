#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""TSTTC experiment with all-frame, bidirectional pair augmentation enabled."""

from .Deep_TTC import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # Six frames produce C(6, 2) = 15 forward temporal pairs.
        self.use_all_frame_pairs = True
        self.frame_pair_sample_num = 0

        # Append the swapped direction immediately after every forward pair.
        # The dataset supplies an explicit scale_gt for the swapped sample so
        # scale(ref->tar) * scale(tar->ref) is exactly 1 before loss clamping.
        self.reverse_aug_append = True
        self.reverse_aug_prob = 1.0
        self.reverse_ttc_mode = "reciprocal_scale"

        # Preserve edge-frame pairs. Any crop area outside the camera image is
        # filled with the same neutral value used by batch-level image padding.
        self.pad_outside_crop = True
        self.crop_padding_value = 127

        # Close the scale interval under inversion. The original [0.65, 1.5]
        # interval becomes [0.65, 1 / 0.65] for bidirectional supervision.
        base_min_scale = self.min_scale
        base_max_scale = self.max_scale
        self.min_scale = min(base_min_scale, 1.0 / base_max_scale)
        self.max_scale = max(base_max_scale, 1.0 / base_min_scale)

        self.exp_name = "Deep_TTC_all_frames_bidirectional"
