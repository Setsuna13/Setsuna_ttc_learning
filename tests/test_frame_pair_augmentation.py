import unittest

import numpy as np
import torch

from data.ttc_dataset import TSTTCDataset, ttc_collate_fn
from data.utils import get_crop_size, get_cropped_imgs
from exp.Deep_TTC_Aug import Exp as AugExp
from model.ttc_head import TTCHead


def make_dataset(reverse_ttc_mode="reciprocal_scale"):
    dataset = TSTTCDataset.__new__(TSTTCDataset)
    dataset.seq_len = 6
    dataset.training = True
    dataset.whole_img = False
    dataset.use_all_frame_pairs = True
    dataset.frame_pair_sample_num = 0
    dataset.reverse_aug_prob = 1.0
    dataset.reverse_aug_append = True
    dataset.reverse_ttc_mode = reverse_ttc_mode
    dataset.frame_rate = 10.0
    dataset.annos = [None]
    return dataset


class FramePairAugmentationTest(unittest.TestCase):
    def test_six_frames_produce_all_fifteen_pairs(self):
        dataset = make_dataset()
        pairs = dataset._get_pair_specs()

        self.assertEqual(len(pairs), 15)
        self.assertEqual(len(set(pairs)), 15)
        self.assertEqual({index for pair in pairs for index in pair}, set(range(6)))
        self.assertTrue(all(ref_pos < tar_pos for ref_pos, tar_pos in pairs))

    def test_forward_and_reverse_samples_are_both_indexed(self):
        dataset = make_dataset()
        sample_index = dataset._build_sample_index()

        self.assertEqual(len(sample_index), 30)
        for offset in range(0, len(sample_index), 2):
            forward = sample_index[offset]
            reverse = sample_index[offset + 1]
            self.assertEqual(forward[:3], reverse[:3])
            self.assertFalse(forward[3])
            self.assertTrue(reverse[3])

    def test_swapped_pair_has_exact_reciprocal_scale_target(self):
        dataset = make_dataset()
        seq = [
            {"ttc_imu": 8.0 - index * 0.1, "img_path": "frame_{}.jpg".format(index)}
            for index in range(6)
        ]

        for ref_pos, tar_pos in dataset._get_pair_specs():
            forward_ref, forward_tar, gap, forward_scale = dataset._load_pair_annos(
                seq, ref_pos, tar_pos, False
            )
            reverse_ref, reverse_tar, reverse_gap, reverse_scale = dataset._load_pair_annos(
                seq, ref_pos, tar_pos, True
            )

            self.assertEqual(gap, reverse_gap)
            self.assertEqual(forward_ref["img_path"], reverse_tar["img_path"])
            self.assertEqual(forward_tar["img_path"], reverse_ref["img_path"])
            self.assertAlmostEqual(forward_scale * reverse_scale, 1.0, places=12)

    def test_collate_preserves_explicit_scale_target_and_pair_metadata(self):
        sample = {
            "imgPair": [
                np.zeros((3, 4, 3), dtype=np.uint8),
                np.ones((3, 4, 3), dtype=np.uint8),
            ],
            "refBoxAnnos": [[0.1, 0.1, 0.8, 0.8]],
            "curBoxAnnos": [[0.2, 0.2, 0.9, 0.9]],
            "ttc_imu": [-4.5],
            "scale_gt": [1.125],
            "curAnnos": [{"anno_id": 1}],
            "frame_gap": [5],
            "pair_indices": [(5, 0)],
            "is_reversed": [True],
            "dynamicRanges": [],
            "masks": [],
        }

        images, annotations, boxes, ttcs = ttc_collate_fn([sample])

        self.assertEqual(tuple(images.shape), (2, 3, 4, 3))
        self.assertEqual(tuple(boxes.shape), (2, 5))
        self.assertTrue(torch.equal(ttcs, torch.tensor([-4.5])))
        self.assertEqual(annotations["scale_gt"], [1.125])
        self.assertEqual(annotations["pair_indices"], [(5, 0)])
        self.assertEqual(annotations["is_reversed"], [True])

    def test_head_prefers_explicit_scale_target(self):
        head = TTCHead.__new__(TTCHead)
        targets = head.prepare_targets(
            gts=torch.tensor([3.0, -3.5]),
            gap=[5, 5],
            scale_gts=[0.8, 1.25],
        )

        self.assertTrue(torch.equal(targets, torch.tensor([0.8, 1.25])))

    def test_edge_crop_is_kept_when_outside_padding_is_enabled(self):
        ref_box = [0.0, 0.4, 0.1, 0.6]
        tar_box = [0.1, 0.3, 0.5, 0.7]

        self.assertIsNone(
            get_crop_size(
                ref_box, tar_box, expand_ratio=1.0, max_scale=1.5,
                allow_outside=False,
            )
        )
        padded_candidate = get_crop_size(
            ref_box, tar_box, expand_ratio=1.0, max_scale=1.5,
            allow_outside=True,
        )

        self.assertIsNotNone(padded_candidate)
        self.assertLess(padded_candidate[0][0], 0.0)

    def test_outside_crop_is_padded_without_numpy_negative_index_wraparound(self):
        image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        ref_box = [-0.25, 0.25, 0.5, 0.75]
        tar_box = [0.25, 0.25, 0.75, 0.75]

        (ref_crop, tar_crop), ref_margins, tar_margins = get_cropped_imgs(
            image,
            image,
            ref_box,
            tar_box,
            receptive_filed=1,
            max_unit_size=500,
            pad_if_needed=True,
            border_value=127,
        )

        self.assertEqual(tuple(ref_crop.shape), (4, 6, 3))
        self.assertEqual(ref_margins, [1, 1, 1, 1])
        self.assertTrue(np.all(ref_crop[:, :2] == 127))
        self.assertTrue(np.array_equal(ref_crop[:, 2:], image[:, :4]))
        self.assertEqual(tar_margins, [1, 1, 1, 1])
        self.assertGreater(tar_crop.size, 0)

    def test_augmentation_experiment_enables_crop_padding(self):
        exp = AugExp()

        self.assertTrue(exp.pad_outside_crop)
        self.assertEqual(exp.crop_padding_value, 127)


if __name__ == "__main__":
    unittest.main()
