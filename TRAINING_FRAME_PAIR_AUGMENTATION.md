# Frame-Pair TTC Augmentation Training

This note records the 6-frame pair augmentation added for TTC training.

## Goal

The original training path uses only the first and last frame in a 6-frame sequence:

```text
frame 0 -> frame 5
```

The augmentation uses all two-frame pairs from the same 6-frame sequence and appends the swapped direction with an exactly reciprocal scale label.

## Frame Pairs

With `sequence_len=6`, forward unordered pairs are:

```text
C(6, 2) = 15
```

Example:

```text
0->1, 0->2, 0->3, 0->4, 0->5
1->2, 1->3, 1->4, 1->5
2->3, 2->4, 2->5
3->4, 3->5
4->5
```

When reverse pairs are appended with probability 1, the training set contains both directions:

```text
15 forward + 15 reverse = 30 = A(6, 2)
```

## Reverse Label Rule

Let `g` be the frame gap, `f = 10 / g` the pair-specific FPS, and `T` the TTC annotation at the forward target frame. The original scale label is:

```text
s_forward = TTC_to_scale(T, f)
```

After swapping the images and boxes:

```text
s_reverse = 1 / s_forward
```

The dataset passes `scale_gt` directly to the head, so `s_forward * s_reverse == 1` before any loss-range clamping. A compatible reverse TTC value is also back-computed for TTC-aware auxiliary losses and logging. Merely changing the sign of TTC is not generally equivalent to scale inversion.

## Label Modes

| Mode | TTC behavior | Scale behavior | Recommended use |
| --- | --- | --- | --- |
| `reciprocal_scale` | TTC is back-computed | Scale is exactly `1 / original_scale` | Bidirectional training |
| `sign` | `T -> -T` | Recomputed from sign-flipped TTC | Legacy ablation only |

`exp/Deep_TTC_Aug.py` selects `reciprocal_scale` and expands the scale-bin interval so it is closed under inversion.

## Recommended Command

Use all 6-frame pairs and keep both directions:

```shell
python tools/train.py \
  -f ./exp/Deep_TTC_Aug.py \
  -d 1 \
  -b 8 \
  --fp16
```

This produces 30 indexed training samples per original 6-frame sequence: 15 temporal pairs times 2 directions. Validation remains unchanged.

The augmented index is computed from the requested sample number instead of
materializing one Python tuple per sample. The logical dataset is still 30
times larger, but its index memory stays essentially constant.

## Epoch Budget

`Deep_TTC_Aug.py` sets `train_epoch_size_multiplier=1.0`. One epoch therefore
contains the same number of samples as the original first-last dataset instead
of becoming 30 times longer. The infinite shuffled sampler continues from its
current position across epoch boundaries:

```text
1 augmented epoch  = 1 x original sequence count
30 augmented epochs = all 30 directed frame pairs covered once
36 augmented epochs = about 1.2 passes over the full augmented index
```

This keeps the optimizer steps, warmup, cosine schedule, checkpoint cadence,
and total runtime comparable to the original 36-epoch configuration while
still guaranteeing that every directed pair is reached during training.

Set `train_epoch_size_multiplier=0` to consume the complete 30-times-larger
index in every epoch. That is substantially more training and is not the
recommended default.

## DataLoader Stability

Training and validation workers are started before CUDA model initialization
and remain persistent. This avoids forking worker processes after CUDA has been
initialized, which can otherwise cause a segmentation fault in
`libtorch_cpu.so`. The augmentation experiment uses four workers by default;
override `data_num_workers` if host RAM or CPU throughput requires a different
value.

## Boundary Crop Retention

`Deep_TTC_Aug.py` enables `pad_outside_crop=true`. When an enlarged ROI crosses an image boundary, the loader now:

1. keeps the requested ROI geometry unchanged;
2. copies the visible pixels into a larger canvas;
3. fills only the out-of-image region with value `127`;
4. shifts the ROI coordinates into that padded canvas.

This replaces the legacy `return None` behavior and avoids NumPy negative-index wraparound. The baseline `Deep_TTC.py` keeps padding disabled for reproduction compatibility.

On a smoke test using 20 real training annotation files (1,505 sequences and 45,150 bidirectional frame-pair samples), the legacy boundary rule rejected 194 samples. Padding retained all 45,150 geometrically valid samples. Corrupt images and non-finite or zero-area annotation boxes are still rejected because they do not contain usable training evidence.

The same behavior can be enabled on the baseline experiment through overrides:

```shell
python tools/train.py \
  -f ./exp/Deep_TTC.py \
  -d 1 \
  -b 8 \
  --fp16 \
  use_all_frame_pairs true \
  reverse_aug_append true \
  reverse_aug_prob 1 \
  reverse_ttc_mode reciprocal_scale
```

Prefer `Deep_TTC_Aug.py`, because it also expands the configured scale interval to cover reciprocal labels without clipping.

## Fixed-Size Alternative

To keep the dataset length unchanged, replace samples in-place instead of appending:

```shell
python tools/train.py \
  -f ./exp/Deep_TTC.py \
  -d 1 \
  -b 8 \
  --fp16 \
  use_all_frame_pairs true \
  reverse_aug_append false \
  reverse_aug_prob 0.5 \
  reverse_ttc_mode reciprocal_scale
```

## Optional Pair Sampling

By default, `frame_pair_sample_num=0` means all 15 forward pairs are used per 6-frame sequence.

To sample fewer forward pairs per sequence:

```shell
frame_pair_sample_num 5
```

If `reverse_aug_append true reverse_aug_prob 1` is also used, those 5 forward pairs become 10 total training pairs after reverse append.

## Validation

Validation remains fixed to the first-last pair, preserving direct comparability with the original protocol and avoiding duplicate annotation IDs in evaluation output.

## Relevant Configs

These options live in `exp/Deep_TTC.py` and are passed to `TSTTCDataset`:

```text
use_all_frame_pairs
frame_pair_sample_num
reverse_aug_prob
reverse_aug_append
reverse_ttc_mode
pad_outside_crop
crop_padding_value
```

Each batch also carries `frame_gap`, `scale_gt`, `pair_indices`, and `is_reversed`. The head prefers the explicit `scale_gt`; older datasets without that field retain the original TTC-to-scale fallback.
