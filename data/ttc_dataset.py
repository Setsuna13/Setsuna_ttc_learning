import copy
import os
import random
import pickle
import itertools
from loguru import logger

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.sampler import Sampler, BatchSampler, SequentialSampler
import torch.distributed as dist
from .utils import get_crop_size, get_cropped_imgs,padding_image_to_same
from .dataset_api import TSTTC

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".JPEG"]
XML_EXT = [".xml"]
PKL_EXT = [".pkl"]
DEFAULT_FRAME_RATE = 10.0


def load_pickle(f):
    return pickle.load(open(f, 'rb'))

def get_file_list(path, type_list):
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in type_list:
                file_names.append(apath)
    return file_names

def scale_ratio_to_ttc(scale_ratio, fps=10):
    ttc = 1 / ((fps * (1 / scale_ratio - 1)) + 1e-9)
    return ttc

def ttc_to_scale_ratio(ttc, fps=10):
    scale_ratio = 1 / ((1 / (ttc * fps) + 1) + 1e-6)
    return scale_ratio


def remove_useless_info(tsttc,first_last=True,seq_len=6):
    if isinstance(tsttc, TSTTC):
        annos = tsttc.annos
        if first_last:
            for key,val in annos.items():
                #set others to None except -1 and -seq_len
                for i in range(-len(val),-1):
                    if i != -seq_len: val[i] = None

            for key,val in tsttc.frameSeqs.items():
                for seqs in val:
                    for i in range(-len(val), -1):
                        if i != -seq_len: val[i] = None


class TSTTCDataset(torchDataset):
    def __init__(
            self,
            data_path = '',
            anno_path = '',
            img_size=(576, 1024),
            preproc=None,
            seq_len=6,
            first_last=True,
            training=True,
            whole_img = False,
            box_downsample_thresh = 300,
            receptive_filed = None,
            min_size_after_padding = 300,
            training_data_ratio = 1.0,
            expand_ratio = 1.1,
            default_max_scale = 1.25,
            frame_gap_scale_ranges=None,
            nerf_path=None,
            nerf_seqs = 0,
            nerf_seed = 0,
            use_all_frame_pairs=False,
            frame_pair_sample_num=0,
            reverse_aug_prob=0.0,
            reverse_aug_append=False,
            reverse_ttc_mode="sign",
            frame_rate=DEFAULT_FRAME_RATE,
            pad_outside_crop=False,
            crop_padding_value=127,
            use_robust_box_crop=False,
            robust_box_occ_thresh=0.3,
            robust_box_area_ratio_thresh=1.8,
            robust_box_height_ratio_thresh=1.6,
            robust_box_center_shift_thresh=0.0,
            **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.anno_path = anno_path
        self.nerf_path = nerf_path
        self.img_size = img_size
        self.preproc = preproc
        self.first_last = first_last
        self.training = training
        self.seq_len = seq_len
        self.use_all_frame_pairs = bool(use_all_frame_pairs or (training and not first_last))
        self.frame_pair_sample_num = int(frame_pair_sample_num)
        self.reverse_aug_prob = float(reverse_aug_prob)
        self.reverse_aug_append = bool(reverse_aug_append)
        self.reverse_ttc_mode = str(reverse_ttc_mode)
        if self.reverse_ttc_mode not in {"sign", "reciprocal_scale"}:
            raise ValueError(
                "reverse_ttc_mode must be 'sign' or 'reciprocal_scale', got {}".format(
                    self.reverse_ttc_mode
                )
            )
        self.frame_rate = float(frame_rate)
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive, got {}".format(self.frame_rate))
        self.pad_outside_crop = bool(pad_outside_crop)
        self.crop_padding_value = crop_padding_value
        self.use_robust_box_crop = bool(use_robust_box_crop)
        self.robust_box_occ_thresh = float(robust_box_occ_thresh)
        self.robust_box_area_ratio_thresh = float(robust_box_area_ratio_thresh)
        self.robust_box_height_ratio_thresh = float(robust_box_height_ratio_thresh)
        self.robust_box_center_shift_thresh = float(robust_box_center_shift_thresh)
        logger.info("Loading TSTTC dataset from {}...".format(self.data_path))
        self.tsttc = TSTTC(self.data_path, self.anno_path,
                           sequence_len=seq_len)
        self.whole_img = whole_img
        self.box_downsample_thresh = box_downsample_thresh
        self.min_size_after_padding = min_size_after_padding
        self.receptive_filed = receptive_filed
        self.anno_ids = self.tsttc.getAnnoIds()
        if training_data_ratio < 1.0:
            if training:
                random.shuffle(self.anno_ids)
            else:
                random.Random(0).shuffle(self.anno_ids)
            self.anno_ids = self.anno_ids[:max(1, int(len(self.anno_ids) * training_data_ratio))]
        self.annos = self.tsttc.loadAnnos(self.anno_ids)
        self.img_ids = self.tsttc.getImgSeqIds()
        self.imgSeqsAnnos = self.tsttc.loadImgSeqs(self.img_ids)
        if self.nerf_path is not None:
            #note: only support box level training!
            logger.info("Loading TSTTC dataset from {}...".format(self.nerf_path))
            self.nerfttc = TSTTC(self.nerf_path)
            nerf_ids = self.nerfttc.getAnnoIds()
            random.seed(nerf_seed)
            random.shuffle(nerf_ids)
            self.nerf_ids = nerf_ids[:nerf_seqs]
            self.annos = self.annos + self.nerfttc.loadAnnos(self.nerf_ids)

        self.expand_ratio = expand_ratio
        self.default_max_scale = default_max_scale
        self.frame_gap_scale_ranges = self._validate_frame_gap_scale_ranges(
            frame_gap_scale_ranges
        )
        self.grid_size = kwargs.get('grid_size',50)
        self.base_sequence_count = len(self.annos)
        self._compact_pair_specs = None
        self._compact_directions_per_pair = 1
        self._sample_count = 0
        self.sample_index = self._build_sample_index()

    @staticmethod
    def _validate_frame_gap_scale_ranges(scale_ranges):
        if scale_ranges is None or len(scale_ranges) == 0:
            return ()
        validated = []
        for frame_gap, bounds in enumerate(scale_ranges, start=1):
            if len(bounds) != 2:
                raise ValueError(
                    "frame-gap {} scale range must contain [min, max]".format(
                        frame_gap
                    )
                )
            min_scale, max_scale = float(bounds[0]), float(bounds[1])
            if not np.isfinite(min_scale) or not np.isfinite(max_scale):
                raise ValueError("frame-gap scale ranges must be finite")
            if min_scale <= 0 or max_scale <= min_scale:
                raise ValueError(
                    "invalid frame-gap {} scale range: {}".format(
                        frame_gap, bounds
                    )
                )
            validated.append((min_scale, max_scale))
        return tuple(validated)

    def _max_scale_for_gap(self, frame_gap):
        if not self.frame_gap_scale_ranges:
            return self.default_max_scale
        frame_gap = int(frame_gap)
        if frame_gap < 1 or frame_gap > len(self.frame_gap_scale_ranges):
            raise ValueError(
                "frame gap {} has no configured scale range".format(frame_gap)
            )
        return self.frame_gap_scale_ranges[frame_gap - 1][1]

    def _build_sample_index(self):
        if self.whole_img:
            return []
        rng = random.Random(0)
        pair_specs = self._get_pair_specs()
        samples_pairs = 0 < self.frame_pair_sample_num < len(pair_specs)
        stochastic_append = (
            self.training
            and self.reverse_aug_append
            and 0 < self.reverse_aug_prob < 1
        )
        if not samples_pairs and not stochastic_append:
            self._compact_pair_specs = pair_specs
            if (
                self.training
                and self.reverse_aug_append
                and self.reverse_aug_prob >= 1
            ):
                self._compact_directions_per_pair = 2
            self._sample_count = (
                len(self.annos)
                * len(pair_specs)
                * self._compact_directions_per_pair
            )
            return None

        sample_index = []
        for anno_idx in range(len(self.annos)):
            cur_pair_specs = pair_specs
            if samples_pairs:
                cur_pair_specs = rng.sample(pair_specs, self.frame_pair_sample_num)
            for ref_pos, cur_pos in cur_pair_specs:
                sample_index.append((anno_idx, ref_pos, cur_pos, False))
                if (
                    self.training
                    and self.reverse_aug_append
                    and self.reverse_aug_prob > 0
                    and rng.random() < self.reverse_aug_prob
                ):
                    sample_index.append((anno_idx, ref_pos, cur_pos, True))
        self._sample_count = len(sample_index)
        return sample_index

    def _get_pair_specs(self):
        if self.use_all_frame_pairs:
            return [
                (ref_pos, cur_pos)
                for ref_pos in range(self.seq_len - 1)
                for cur_pos in range(ref_pos + 1, self.seq_len)
            ]
        return [(0, self.seq_len - 1)]

    def _get_indexed_pair(self, index):
        if self.sample_index is not None:
            anno_idx, ref_pos, cur_pos, reverse_pair = self.sample_index[index]
        else:
            pair_index, direction = divmod(
                index, self._compact_directions_per_pair
            )
            anno_idx, pair_offset = divmod(
                pair_index, len(self._compact_pair_specs)
            )
            ref_pos, cur_pos = self._compact_pair_specs[pair_offset]
            reverse_pair = bool(direction)
        if (
            self.training
            and not self.reverse_aug_append
            and self.reverse_aug_prob > 0
            and random.random() < self.reverse_aug_prob
        ):
            reverse_pair = not reverse_pair
        return anno_idx, ref_pos, cur_pos, reverse_pair

    @staticmethod
    def _get_ttc(obj_anno):
        try:
            return obj_anno['ttc_imu']
        except Exception:
            return obj_anno.ttc_imu

    @staticmethod
    def _set_ttc(obj_anno, value):
        try:
            obj_anno['ttc_imu'] = value
        except Exception:
            obj_anno.ttc_imu = value

    def _reverse_ttc(self, ttc, frame_gap):
        if self.reverse_ttc_mode == "reciprocal_scale":
            fps = self.frame_rate / float(frame_gap)
            scale_ratio = ttc_to_scale_ratio(ttc, fps=fps)
            reversed_scale_ratio = 1.0 / scale_ratio
            return scale_ratio_to_ttc(reversed_scale_ratio, fps=fps)
        return -ttc

    def _forward_pair_scale(self, seq, cur_pos, frame_gap):
        """Return the original ref->target scale label for one temporal pair."""
        fps = self.frame_rate / float(frame_gap)
        return float(ttc_to_scale_ratio(self._get_ttc(seq[cur_pos]), fps=fps))

    def _load_pair_annos(self, seq, ref_pos, cur_pos, reverse_pair):
        if ref_pos >= len(seq) or cur_pos >= len(seq):
            raise IndexError("frame pair ({}, {}) out of sequence length {}".format(ref_pos, cur_pos, len(seq)))
        frame_gap = abs(cur_pos - ref_pos)
        forward_scale = self._forward_pair_scale(seq, cur_pos, frame_gap)
        if reverse_pair:
            objAnnoRef = copy.deepcopy(seq[cur_pos])
            objAnnoCur = copy.deepcopy(seq[ref_pos])
            original_ttc = self._get_ttc(seq[cur_pos])
            reversed_ttc = self._reverse_ttc(original_ttc, frame_gap)
            self._set_ttc(objAnnoCur, reversed_ttc)
            if self.reverse_ttc_mode == "reciprocal_scale":
                # Pass this value directly to the head. Re-encoding it through TTC would
                # introduce conversion epsilon and break the exact inverse relationship.
                scale_gt = 1.0 / forward_scale
            else:
                fps = self.frame_rate / float(frame_gap)
                scale_gt = float(ttc_to_scale_ratio(reversed_ttc, fps=fps))
        else:
            objAnnoRef = copy.deepcopy(seq[ref_pos])
            objAnnoCur = copy.deepcopy(seq[cur_pos])
            scale_gt = forward_scale
        return objAnnoRef, objAnnoCur, frame_gap, scale_gt

    @staticmethod
    def _box_array(obj_anno):
        try:
            box = obj_anno['box2d']
        except Exception:
            box = obj_anno.box2d
        return np.asarray(box, dtype=np.float32)

    @staticmethod
    def _box_occ_ratio(obj_anno):
        try:
            return float(obj_anno.get('occ_ratio', 0.0))
        except Exception:
            return float(getattr(obj_anno, 'occ_ratio', 0.0))

    @staticmethod
    def _box_stats(box):
        w = max(float(box[2] - box[0]), 1e-6)
        h = max(float(box[3] - box[1]), 1e-6)
        cx = float((box[0] + box[2]) * 0.5)
        cy = float((box[1] + box[3]) * 0.5)
        return w, h, w * h, cx, cy

    @staticmethod
    def _sym_ratio(a, b):
        a = max(float(a), 1e-6)
        b = max(float(b), 1e-6)
        return max(a / b, b / a)

    def _is_box_suspicious(self, seq, pos):
        if not self.use_robust_box_crop:
            return False
        if self._box_occ_ratio(seq[pos]) > self.robust_box_occ_thresh:
            return True

        boxes = [self._box_array(obj) for obj in seq]
        ref_boxes = [
            box for idx, box in enumerate(boxes)
            if idx != pos and self._box_occ_ratio(seq[idx]) <= self.robust_box_occ_thresh
        ]
        if len(ref_boxes) == 0:
            return False

        ref_box = np.median(np.stack(ref_boxes, axis=0), axis=0)
        cur_w, cur_h, cur_area, cur_cx, cur_cy = self._box_stats(boxes[pos])
        ref_w, ref_h, ref_area, ref_cx, ref_cy = self._box_stats(ref_box)
        if self._sym_ratio(cur_area, ref_area) > self.robust_box_area_ratio_thresh:
            return True
        if self._sym_ratio(cur_h, ref_h) > self.robust_box_height_ratio_thresh:
            return True
        if self.robust_box_center_shift_thresh > 0:
            ref_diag = max((ref_w ** 2 + ref_h ** 2) ** 0.5, 1e-6)
            center_shift = (((cur_cx - ref_cx) ** 2 + (cur_cy - ref_cy) ** 2) ** 0.5) / ref_diag
            if center_shift > self.robust_box_center_shift_thresh:
                return True
        return False

    def _robust_box_for_crop(self, seq, pos):
        box = self._box_array(seq[pos])
        if not self._is_box_suspicious(seq, pos):
            return box

        good_positions = [
            idx for idx in range(len(seq))
            if idx != pos and not self._is_box_suspicious(seq, idx)
        ]
        if len(good_positions) == 0:
            return box

        left_positions = [idx for idx in good_positions if idx < pos]
        right_positions = [idx for idx in good_positions if idx > pos]
        left = max(left_positions) if left_positions else None
        right = min(right_positions) if right_positions else None
        if left is not None and right is not None:
            alpha = float(pos - left) / max(float(right - left), 1.0)
            left_box = self._box_array(seq[left])
            right_box = self._box_array(seq[right])
            return left_box * (1.0 - alpha) + right_box * alpha
        if left is not None:
            return self._box_array(seq[left])
        return self._box_array(seq[right])

    def _pair_boxes_for_crop(self, seq, ref_pos, cur_pos, reverse_pair, objAnnoRef, objAnnoCur):
        if not self.use_robust_box_crop:
            return self._box_array(objAnnoRef), self._box_array(objAnnoCur)
        if reverse_pair:
            ref_box = self._robust_box_for_crop(seq, cur_pos)
            cur_box = self._robust_box_for_crop(seq, ref_pos)
        else:
            ref_box = self._robust_box_for_crop(seq, ref_pos)
            cur_box = self._robust_box_for_crop(seq, cur_pos)
        return ref_box, cur_box

    def __len__(self):
        if self.whole_img:
            return len(self.tsttc.frameSeqs)
        return self._sample_count

    def resize_img(self, img,):
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img
    def pull_item(self, index):
        result_dict = {'imgPair':[],'refBoxAnnos':[],'curBoxAnnos':[],'ttc_imu':[],\
                       'scale_gt':[],'curAnnos':[],'dynamicRanges':[],'frame_gap':[],
                       'pair_indices':[],'is_reversed':[],'masks':[]}
        if self.first_last or not self.whole_img:
            cur_idx,ref_idx = -1,0
        else: # whole-image all-pair mode is not supported; keep the legacy first/last pair.
            cur_idx,ref_idx = -1,0
        if self.whole_img:
            frameSeq = self.tsttc.frameSeqs[index]
            for i in range(len(frameSeq)):
                try:
                    objAnnoRef,objAnnoCur =  frameSeq[i][-self.seq_len:][ref_idx],frameSeq[i][-self.seq_len:][cur_idx]
                except Exception as e:
                    print('load: ',frameSeq[i])
                    logger.warning('fail to load image pair: %s' % (e))
                    return result_dict

                if 'ttc_imu' not in objAnnoCur:
                    logger.warning('no ttc imu: %s' % (objAnnoCur))
                    return result_dict
                if i == 0:
                    try:
                        imgRef = self.resize_img(cv2.imread(objAnnoRef['img_path']))
                        imgCur = self.resize_img(cv2.imread(objAnnoCur['img_path']))
                    except AttributeError:
                        logger.warning(
                            'fail to load image pair: %s or %s' % (objAnnoRef['img_path'], objAnnoCur['img_path']))
                        return result_dict
                    result_dict['imgPair'].extend([imgRef,imgCur])
                frame_gap = self.seq_len-ref_idx-1
                max_scale = self._max_scale_for_gap(frame_gap)
                seq = frameSeq[i][-self.seq_len:]
                ref_box, cur_box = self._pair_boxes_for_crop(seq, 0, self.seq_len - 1, False, objAnnoRef, objAnnoCur)
                candidate_boxes = get_crop_size(
                    ref_box,
                    cur_box,
                    max_scale=max_scale,
                    expand_ratio=self.expand_ratio,
                    allow_outside=self.pad_outside_crop,
                )
                if candidate_boxes is not None:
                    result_dict['refBoxAnnos'].append(candidate_boxes[0])
                    result_dict['curBoxAnnos'].append(candidate_boxes[3])
                    result_dict['ttc_imu'].append(objAnnoCur['ttc_imu'])
                    fps = self.frame_rate / float(frame_gap)
                    result_dict['scale_gt'].append(float(ttc_to_scale_ratio(objAnnoCur['ttc_imu'], fps=fps)))
                    result_dict['curAnnos'].append(objAnnoCur)
                    result_dict['frame_gap'].append(frame_gap)
                    result_dict['pair_indices'].append((0, self.seq_len - 1))
                    result_dict['is_reversed'].append(False)
        else:
            result_dict['min_size_after_padding'] = self.min_size_after_padding
            anno_idx, ref_pos, cur_pos, reverse_pair = self._get_indexed_pair(int(index))
            try:
                seq = self.annos[anno_idx][-self.seq_len:]
                objAnnoRef, objAnnoCur, frame_gap, scale_gt = self._load_pair_annos(
                    seq, ref_pos, cur_pos, reverse_pair
                )
            except IndexError:
                logger.warning('fail to load image pair: %s' % index)
                return result_dict
            try:
                imgRef = self.resize_img(cv2.imread(objAnnoRef['img_path']))
                imgCur = self.resize_img(cv2.imread(objAnnoCur['img_path']))
            except Exception as e:
                logger.warning('fail to load image pair: %s or %s'%(objAnnoRef['img_path'],objAnnoCur['img_path']))
                return result_dict

            max_scale = self._max_scale_for_gap(frame_gap)
            ref_box, cur_box = self._pair_boxes_for_crop(seq, ref_pos, cur_pos, reverse_pair, objAnnoRef, objAnnoCur)
            candidate_boxes = get_crop_size(
                ref_box,
                cur_box,
                max_scale=max_scale,
                expand_ratio=self.expand_ratio,
                allow_outside=self.pad_outside_crop,
            )
            if candidate_boxes is not None:
                result_dict['refBoxAnnos'].append(candidate_boxes[0])
                result_dict['curBoxAnnos'].append(candidate_boxes[3])
                result_dict['ttc_imu'].append(objAnnoCur['ttc_imu'])
                result_dict['scale_gt'].append(scale_gt)
                result_dict['curAnnos'].append(objAnnoCur)
                result_dict['imgPair'], result_dict['ref_padding'], result_dict['cur_padding'] = get_cropped_imgs(imgRef, imgCur, result_dict['refBoxAnnos'][0],
                                                                result_dict['curBoxAnnos'][0], self.receptive_filed,
                                                                max_unit_size=self.box_downsample_thresh,
                                                                pad_if_needed=self.pad_outside_crop,
                                                                border_value=self.crop_padding_value,
                                                                )
                result_dict['frame_gap'].append(frame_gap)
                if reverse_pair:
                    result_dict['pair_indices'].append((cur_pos, ref_pos))
                else:
                    result_dict['pair_indices'].append((ref_pos, cur_pos))
                result_dict['is_reversed'].append(reverse_pair)

            else:
                logger.warning('box out of img after enlarging: %s' % objAnnoCur['img_path'])
        return result_dict

    def __getitem__(self, index):
        result_dict = self.pull_item(int(index))
        if self.preproc is not None and len(result_dict['refBoxAnnos']) > 0:
            result_dict['imgPair'][0],result_dict['refBoxAnnos'] = self.preproc(result_dict['imgPair'][0],np.array(result_dict['refBoxAnnos']),self.img_size)
            result_dict['imgPair'][1],result_dict['curBoxAnnos'] = self.preproc(result_dict['imgPair'][1],np.array(result_dict['curBoxAnnos']),self.img_size)
        return result_dict
class TTCDataset(torchDataset):
    '''
    TTC sequence
    '''

    def __init__(
            self,
            data_path='',
            img_size=(576, 1024),
            preproc=None,
            seq_len=5,
            first_last=True,
            training=True,
            debug_flag=True,
            add_affine=False,
            use_nerf = False,
            resample = False,
            expand_ratio = 1.1,
            nerf_data_path = '',
            nerf_ratio = 0.1,
            nerf_seed = 0,
            max_scale = 1.5,
            tsttc = None
    ):
        super().__init__()
        self.data_path = data_path
        self.img_size = img_size
        self.preproc = preproc
        self.first_last = first_last
        self.training = training
        self.affine = add_affine
        self.seq_len = seq_len
        self.use_nerf = use_nerf
        self.resample = resample
        self.nerf_data_path = nerf_data_path
        self.nerf_seed = nerf_seed
        self.nerf_ratio = nerf_ratio
        self.expand_ratio = expand_ratio
        self.default_max_scale = max_scale
        #all_anno, seqs = self.reformat_anno(seq_len, self.data_path)

        #load dataset
        if tsttc:
            self.tsttc = tsttc
            condition_ids = self.tsttc.getAnnoIds(cam_ids=[1,3,4,8,9])
            seqs = self.tsttc.loadAnnos(condition_ids)
            seqs = [seq[-seq_len:] for seq in seqs]
        else:
            if type(data_path) is str:
                all_anno, seqs = self.reformat_anno(seq_len,self.data_path)
            else:
                seqs = []
                print(self.data_path)
                for tmp_pth in self.data_path:
                    _anno, _seqs = self.reformat_anno(seq_len,tmp_pth)
                    seqs = seqs + _seqs
        # use nerf data
        if self.use_nerf and self.nerf_ratio > 0 and self.training:
            _,seqs_nerf = self.reformat_anno(seq_len,path=self.nerf_data_path,nerf=True)
            if nerf_ratio<1:
                seqs_nerf = seqs_nerf[:int(nerf_ratio*len(seqs))]
            else:
                seq_03,seq_36 = [],[]
                for seq in seqs_nerf:
                    if 0<seq[-1].ttc_imu<=3 and len(seq_03)<nerf_ratio:seq_03.append(seq)
                    elif 3<seq[-1].ttc_imu<=6 and len(seq_36)<nerf_ratio:seq_36.append(seq)
                seqs_nerf = seq_03+seq_36
            seqs = seqs + seqs_nerf

        if training:
            random.seed(42)
            random.shuffle(seqs)
            self.seqs = seqs
        else:
            self.seqs = seqs
            self.first_last = True

    def __len__(self):
        return len(self.seqs)

    def reformat_anno(self, seq_len=5,path = '',nerf = False):
        if nerf:
            anno_names = get_file_list(path, PKL_EXT)
            random.seed(42-self.nerf_seed)
            random.shuffle(anno_names)
        else:
            anno_names = get_file_list(path, PKL_EXT)
        anno_list = []
        input_list = []

        for anno_name in anno_names:  # bag level
            contents = load_pickle(anno_name)
            if contents == []: continue
            for content in contents:
                bag_stamp = content[-1].bag_stamp.split('/')[-1]
                img_name = str(content[-1].ts) + '.jpg'
                cam = 'cam' + str(content[-1].cam_id)
                img_path = os.path.join(path, bag_stamp, cam, img_name)

                ttc_imu = content[-1]['ttc_imu']
                if not os.path.exists(img_path):
                    print('file not exist:', img_path)
                    continue
                for frame_idx in range(len(content)) :
                    content[frame_idx].bag_stamp =  os.path.join(path, bag_stamp)
                    content[frame_idx].img_path = os.path.join(path, bag_stamp, cam, str(content[frame_idx].ts) + '.jpg')
                input_list.append(content[-seq_len:])

        return anno_list, input_list

    def pull_item(self, seq):
        imgs = []
        all_boxes = []
        ttc_gts_dict = {}
        if self.first_last:
            seq = [seq[0], seq[-1]]
            ttc_gts_dict['gap'] = self.seq_len-1
        else: #random gap
            frame_gap = random.randint(1,self.seq_len-1)
            seq = [seq[-1-frame_gap], seq[-1]]
            ttc_gts_dict['gap'] = frame_gap
        for img_annos in seq:
            cam = 'cam' + str(img_annos.cam_id)
            img_name = str(img_annos.ts) + '.jpg'
            img_path = os.path.join(img_annos.bag_stamp, cam, img_name)
            img = cv2.imread(img_annos.img_path)
            height, width = img.shape[:2]
            img_info = (height, width)
            r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
            img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            boxes = np.array([img_annos.box2d])
            imgs.append(img)
            all_boxes.append(boxes)
        ttc_gt,_gt_annos = [seq[-1].ttc_imu], [seq[-1]]
        enlarge_boxes,ttc_gts,gt_annos = [],[],[]
        for i in range(all_boxes[0].shape[0]):# for each box pair
            #fecth the box pair
            res = get_crop_size(all_boxes[0][i], all_boxes[1][i],expand_ratio=self.expand_ratio,max_scale=self.default_max_scale)
            if res is not None:
                enlarge_bbox, enlarge_cur_bbox, bbox, cur_bbox = res
                enlarge_boxes.append([enlarge_bbox, bbox, cur_bbox])
                ttc_gts.append(ttc_gt[i])
                gt_annos.append(_gt_annos[i])
            else:
                print('expand box fail:!!!!')
        ttc_gts_dict['ttc_gts'] = ttc_gts
        if self.training:
            return imgs, all_boxes, enlarge_boxes, ttc_gts_dict
        else:
            return imgs, all_boxes, enlarge_boxes, ttc_gts_dict, gt_annos

    def __getitem__(self, item):
        seq = self.seqs[item]
        if self.training:
            imgs, targets, enlarge_boxes, ttc_gts_dict = self.pull_item(seq)
        else:
            imgs, targets, enlarge_boxes, ttc_gts_dict, gt_annos = self.pull_item(seq)
        if self.preproc is not None:
            _imgs, _targets = [], []
            for i in range(len(imgs)):
                if self.affine:#extra augmentation
                    if i != len(imgs) - 1:
                        enlarge_boxes = np.array(enlarge_boxes)
                        enlarge_boxes = enlarge_boxes.reshape([-1, 4])
                        img, enlarge_boxes = self.preproc(imgs[i], enlarge_boxes, self.img_size, item)
                        enlarge_boxes = enlarge_boxes.reshape([-1, 3, 4])
                        enlarge_boxes = enlarge_boxes.tolist()
                    else:
                        img, _ = self.preproc(imgs[i], [], self.img_size, item)
                else:
                    img, _ = self.preproc(imgs[i], targets[i], self.img_size)
                _imgs.append(img)
            _targets = targets
        else:
            _imgs, _targets = imgs, targets
        if self.training:
            return _imgs, _targets, enlarge_boxes, ttc_gts_dict
        else:
            return _imgs, _targets, enlarge_boxes, ttc_gts_dict, gt_annos


class TrainSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


class TestSampler(SequentialSampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class TTCBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size

def ttc_collate_fn(batch):
    cur_valid_idx = 0
    imgs, boxes, ttcs, padding_sizes = [], [], [], []
    dictAnnos = {
        'metaAnnos': [],
        'dynamicRanges': [],
        'frame_gap': [],
        'scale_gt': [],
        'pair_indices': [],
        'is_reversed': [],
        'masks': []
    }
    for sample in batch:
        if len(sample['refBoxAnnos']):
            imgs.extend(sample['imgPair'])
            ttcs.extend(sample['ttc_imu'])
            dictAnnos['metaAnnos'].extend(sample['curAnnos'])
            dictAnnos['frame_gap'].extend(sample['frame_gap'])
            dictAnnos['scale_gt'].extend(sample.get('scale_gt', []))
            dictAnnos['pair_indices'].extend(sample.get('pair_indices', []))
            dictAnnos['is_reversed'].extend(sample.get('is_reversed', []))
            if 'dynamicRanges' in sample:
                dictAnnos['dynamicRanges'].extend(sample['dynamicRanges'])
            if 'masks' in sample:
                dictAnnos['masks'].extend(sample['masks'])
            roi_prefix_ref = torch.ones([len(sample['refBoxAnnos']), 1], dtype=torch.float32) * (cur_valid_idx)
            roi_prefix_cur = torch.ones([len(sample['curBoxAnnos']), 1], dtype=torch.float32) * (cur_valid_idx)
            roi_box_ref = torch.cat([roi_prefix_ref, torch.tensor(sample['refBoxAnnos'])], dim=1)
            roi_box_cur = torch.cat([roi_prefix_cur, torch.tensor(sample['curBoxAnnos'])], dim=1)
            roi_boxes = torch.stack([roi_box_ref, roi_box_cur], dim=0).permute(1, 0, 2).flatten(0, 1)
            if 'min_size_after_padding' in sample:
                padding_sizes.append(sample['ref_padding'])
                padding_sizes.append(sample['cur_padding'])
            boxes.append(roi_boxes)
            cur_valid_idx += 1
    if len(boxes)==0:
        return None,None,None,None
    boxes = torch.cat(boxes, dim=0)
    if len(padding_sizes):#only box area
        default_padding_size = [sample['min_size_after_padding'],sample['min_size_after_padding']]
        imgs, orininal_box_list = padding_image_to_same(imgs, padding_sizes, dst_size=default_padding_size)
        normed_orininal_boxes = torch.tensor(orininal_box_list,dtype=torch.float32)
        H,W = imgs[0].shape[1:]
        roi_idx = boxes[:,:1]
        normed_orininal_boxes[:,1::2] = normed_orininal_boxes[:,1::2]/H
        normed_orininal_boxes[:,0::2] = normed_orininal_boxes[:,0::2]/W
        normed_orininal_boxes = torch.cat([roi_idx,normed_orininal_boxes],dim=1)
        boxes = normed_orininal_boxes
    # TODO del None in final version
    if len(imgs) == 0:
        tensor_imgs= None
    else:
        tensor_imgs = [torch.tensor(img) for img in imgs]
        if len(dictAnnos['dynamicRanges']):
            dictAnnos['dynamicRanges'] =[torch.tensor(tmpRange) for tmpRange in dictAnnos['dynamicRanges']]
            dictAnnos['dynamicRanges'] = torch.stack(dictAnnos['dynamicRanges'],dim=0)
        if len(dictAnnos['masks']):
            dictAnnos['masks'] = torch.stack(dictAnnos['masks'],dim=0)
    return torch.stack(tensor_imgs, dim=0),dictAnnos, boxes, torch.tensor(ttcs)



def collate_fn(batch):
    tar = []
    imgs = []
    enlarge_boxes = []
    ttc_gts, ttc_gaps = [], []
    ttc_gts_dict = {}
    for sample in batch:
        tar_ori, ttc_ori, ttc_gap = [], [], []

        for img in sample[0]:
            imgs.append(torch.tensor(img))
        for boxes in sample[1]:
            tar_ori.append(torch.tensor(boxes))

        ttc_ori.append(torch.tensor(sample[3]['ttc_gts']))
        ttc_gap.append(torch.tensor(sample[3]['gap']))
        tar.extend(tar_ori)
        enlarge_boxes.append(sample[2])
        ttc_gts.extend(ttc_ori)
        ttc_gaps.extend(ttc_gap)
    ttc_gts_dict['ttc_gts'], ttc_gts_dict['gap'] = ttc_gts, ttc_gaps
    return torch.stack(imgs), tar, enlarge_boxes, ttc_gts_dict

def collate_fn_eval(batch):
    tar = []
    imgs = []
    enlarge_boxes = []
    ttc_gts,ttc_gaps = [],[]
    annos = []
    ttc_gts_dict = {}
    for sample in batch:
        tar_ori, ttc_ori, ttc_gap = [], [], []

        for img in sample[0]:
            imgs.append(torch.tensor(img))
        for boxes in sample[1]:
            tar_ori.append(torch.tensor(boxes))

        ttc_ori.append(torch.tensor(sample[3]['ttc_gts']))
        ttc_gap.append(torch.tensor(sample[3]['gap']))

        tar.extend(tar_ori)
        enlarge_boxes.append(sample[2])
        ttc_gts.extend(ttc_ori)
        ttc_gaps.extend(ttc_gap)
        annos.extend(sample[4])
    ttc_gts_dict['ttc_gts'], ttc_gts_dict['gap'] = ttc_gts, ttc_gaps
    return torch.stack(imgs), tar, enlarge_boxes, ttc_gts_dict, annos

def get_train_loader(batch_size, data_num_workers, dataset,sequence_flag=False):
    fn = collate_fn
    sampler = TTCBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': fn
    }
    ttc_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return ttc_loader


def get_eval_loader(batch_size, data_num_workers, dataset,sequence_flag=False):
    fn = collate_fn_eval
    sampler = TTCBatchSampler(TestSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': fn
    }
    ttc_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    return ttc_loader

def get_ttc_loader(batchSize,data_num_workers,dataset,is_dist = False,seed=0):
    if dataset.training:
        if seed is None: seed = 0
        InfSampler = InfiniteSampler(len(dataset), seed=seed)
        sampler = TTCBatchSampler(sampler=InfSampler,batch_size=batchSize,drop_last=False)
    else:
        if is_dist:
            sampler = TTCBatchSampler(torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
                                      , batchSize, drop_last=False)
        else:
            sampler = TTCBatchSampler(TestSampler(dataset),batchSize,drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        "collate_fn": ttc_collate_fn
    }
    if data_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    if dataset.training:
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    ttc_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return ttc_loader

def worker_init_reset_seed(worker_id):
    # PyTorch has already assigned a distinct worker seed. Calling
    # torch.manual_seed() here can touch libtorch/CUDA state in a forked worker.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
