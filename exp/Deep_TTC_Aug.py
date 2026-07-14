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

        # Keep the optimizer/scheduler epoch comparable to the original
        # first-last training budget. The infinite sampler still traverses all
        # 30 augmented variants across 30 epochs instead of making every epoch
        # 30 times longer.
        self.train_epoch_size_multiplier = 1.0
        self.data_num_workers = 4

        # Use the original range assigned to each frame gap, then close every
        # interval under inversion for ref/target swapping. A mixed batch gets
        # one 50-bin scale grid per sample instead of sharing the gap-5 grid.
        self.frame_gap_scale_ranges = [
            [min(low, 1.0 / high), max(high, 1.0 / low)]
            for low, high in self.min_max_scale_list
        ]
        self.min_scale = min(bounds[0] for bounds in self.frame_gap_scale_ranges)
        self.max_scale = max(bounds[1] for bounds in self.frame_gap_scale_ranges)

        self.exp_name = "Deep_TTC_all_frames_bidirectional"
