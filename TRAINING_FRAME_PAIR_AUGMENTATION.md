# Frame-Pair TTC Augmentation Training

This note records the 6-frame pair augmentation added for TTC training.

## Goal

The original training path uses only the first and last frame in a 6-frame sequence:

```text
frame 0 -> frame 5
```

The new augmentation can use all two-frame pairs from the same 6-frame sequence and can also append the time-reversed pair with a flipped TTC label.

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

For a forward sample:

```text
ref = early frame
cur = late frame
TTC = -6
```

The appended reverse sample becomes:

```text
ref = late frame
cur = early frame
TTC = +6
```

The image pair and boxes are swapped, `frame_gap` stays the same, and the head converts TTC to scale with `fps = 10 / frame_gap`.

## Label Modes

| Mode | TTC behavior | Scale behavior | Recommended use |
| --- | --- | --- | --- |
| `sign` | `-6 -> +6` | Recomputed from flipped TTC | Main training; preserves TTC sign semantics |
| `reciprocal_scale` | TTC is back-computed | Scale is exactly `1 / original_scale` | Ablation for strict scale inversion |

Use `sign` first. Use `reciprocal_scale` only when the experiment specifically requires exact reciprocal scale labels.

## Recommended Command

Use all 6-frame pairs and keep both original and reversed samples:

```shell
python tools/train.py \
  -f ./exp/Deep_TTC.py \
  -d 1 \
  -b 8 \
  --fp16 \
  use_all_frame_pairs true \
  reverse_aug_append true \
  reverse_aug_prob 1 \
  reverse_ttc_mode sign
```

This trains on the original forward samples and the appended reversed samples.

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
  reverse_ttc_mode sign
```

## Optional Pair Sampling

By default, `frame_pair_sample_num=0` means all 15 forward pairs are used per 6-frame sequence.

To sample fewer forward pairs per sequence:

```shell
frame_pair_sample_num 5
```

If `reverse_aug_append true reverse_aug_prob 1` is also used, those 5 forward pairs become 10 total training pairs after reverse append.

## Validation

Validation remains fixed to the first-last pair. The current evaluator converts predicted scale to TTC with a fixed `fps = 10 / (sequence_len - 1)`, so mixing validation pairs with different frame gaps would make the reported TTC metric inconsistent.

## Relevant Configs

These options live in `exp/Deep_TTC.py` and are passed to `TSTTCDataset`:

```text
use_all_frame_pairs
frame_pair_sample_num
reverse_aug_prob
reverse_aug_append
reverse_ttc_mode
```
