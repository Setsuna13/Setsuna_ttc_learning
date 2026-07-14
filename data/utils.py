from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import numpy as np
import torch


def _is_valid_box(box):
    box = np.asarray(box, dtype=np.float32)
    return (
        box.shape == (4,)
        and np.isfinite(box).all()
        and float(box[2] - box[0]) > 0
        and float(box[3] - box[1]) > 0
    )


def get_crop_size(
    bbox,
    cur_bbox,
    scale_beta=0.1,
    expand_ratio=1.1,
    max_scale=1.26,
    allow_outside=False,
):
    eps = 0.002
    if not _is_valid_box(bbox) or not _is_valid_box(cur_bbox):
        return None

    bbox, cur_bbox = expand_bbox_for_bg(
        bbox, cur_bbox, expand_ratio, allow_outside=allow_outside
    )

    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    cur_bbox_h = cur_bbox[3] - cur_bbox[1]
    cur_bbox_w = cur_bbox[2] - cur_bbox[0]
    cur_center_x = (cur_bbox[2] + cur_bbox[0]) / 2
    cur_center_y = (cur_bbox[3] + cur_bbox[1]) / 2

    scale_ratio = max_scale  # max(1 / min_scale, max_scale)
    crop_h = scale_ratio * cur_bbox_h
    crop_w = scale_ratio * cur_bbox_w

    enlarge_bbox = [center_x - crop_w / 2, center_y - crop_h / 2, center_x + crop_w / 2, center_y + crop_h / 2]
    enlarge_cur_bbox = [cur_center_x - crop_w / 2, cur_center_y - crop_h / 2, cur_center_x + crop_w / 2,
                        cur_center_y + crop_h / 2]

    crop_inside = (
        enlarge_bbox[0] > eps
        and enlarge_bbox[2] + eps < 1
        and enlarge_bbox[1] > eps
        and enlarge_bbox[3] + eps < 1
    )
    if not allow_outside and not crop_inside:
        return None
    return enlarge_bbox, enlarge_cur_bbox, bbox, cur_bbox


def expand_bbox_for_bg(bbox, cur_bbox, ratio, allow_outside=False):
    if not allow_outside:
        bbox_ratio = get_valid_ratio(bbox, ratio)
        last_bbox_ratio = get_valid_ratio(cur_bbox, ratio)
        ratio = min(bbox_ratio, last_bbox_ratio)
    bbox = expand_box(bbox, ratio)
    cur_bbox = expand_box(cur_bbox, ratio)
    return bbox, cur_bbox


def get_valid_ratio(bbox, ratio):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    ctr_x = (x1 + x2) / 2
    ctr_y = (y1 + y2) / 2
    delta_x = (x2 - x1)
    delta_y = (y2 - y1)

    if delta_x <= 0 or delta_y <= 0:
        return 0.0
    max_ratio_x = min(ctr_x, 1 - ctr_x) / (delta_x / 2)
    max_ratio_y = min(ctr_y, 1 - ctr_y) / (delta_y / 2)

    max_ratio = min(max_ratio_x, max_ratio_y)
    bg_ratio = max(0.0, min(ratio, max_ratio))
    return bg_ratio


def expand_box(bbox, ratio):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    ctr_x = (x1 + x2) / 2
    ctr_y = (y1 + y2) / 2
    delta_x = (x2 - x1)
    delta_y = (y2 - y1)
    ndelta_x = delta_x * ratio
    ndelta_y = delta_y * ratio

    nx1 = ctr_x - ndelta_x / 2
    nx2 = ctr_x + ndelta_x / 2
    ny1 = ctr_y - ndelta_y / 2
    ny2 = ctr_y + ndelta_y / 2

    return [nx1, ny1, nx2, ny2]

def crop_bbox_img(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[int(y1):int(y2), int(x1):int(x2)]


def crop_bbox_img_with_padding(img, bbox, context=0, border_value=127):
    """Crop a normalized box without clipping it, padding pixels outside the image."""
    if not _is_valid_box(bbox):
        raise ValueError("invalid crop box: {}".format(bbox))

    height, width = img.shape[:2]
    context = max(0, int(context or 0))
    x1 = int(float(bbox[0]) * width)
    y1 = int(float(bbox[1]) * height)
    x2 = int(float(bbox[2]) * width)
    y2 = int(float(bbox[3]) * height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("crop box is smaller than one pixel: {}".format(bbox))

    crop_x1, crop_y1 = x1 - context, y1 - context
    crop_x2, crop_y2 = x2 + context, y2 + context
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    if img.ndim == 2:
        output_shape = (crop_height, crop_width)
    else:
        output_shape = (crop_height, crop_width, img.shape[2])
    cropped = np.empty(output_shape, dtype=img.dtype)
    cropped[...] = border_value

    src_x1, src_y1 = max(0, crop_x1), max(0, crop_y1)
    src_x2, src_y2 = min(width, crop_x2), min(height, crop_y2)
    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1, dst_y1 = src_x1 - crop_x1, src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        cropped[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    # The collate function reconstructs the ROI as
    # [left, top, crop_width - right, crop_height - bottom].
    roi_margins = [
        x1 - crop_x1,
        y1 - crop_y1,
        crop_x2 - x2,
        crop_y2 - y2,
    ]
    return cropped, roi_margins


def downsample_img(img, rate=0.5):
    _img = cv2.resize(
        img,
        (int(img.shape[1] * rate), int(img.shape[0] * rate)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    return _img


def get_cropped_imgs(
    ref_img,
    tar_img,
    ref_box,
    tar_box,
    receptive_filed=32,
    max_unit_size=500,
    pad_if_needed=False,
    border_value=127,
):
    H, W = ref_img.shape[:2]
    receptive_filed = max(0, int(receptive_filed or 0))
    if pad_if_needed:
        ref, ref_padding = crop_bbox_img_with_padding(
            ref_img, ref_box, context=receptive_filed, border_value=border_value
        )
        tar, tar_padding = crop_bbox_img_with_padding(
            tar_img, tar_box, context=receptive_filed, border_value=border_value
        )
    else:
        tar_padding = [min(int(tar_box[0] * W),receptive_filed),min(int(tar_box[1] * H),receptive_filed),
                       min(int(W-tar_box[2] * W),receptive_filed),min(int(H-tar_box[3] * H),receptive_filed)]
        ref_padding = [min(int(ref_box[0] * W), receptive_filed), min(int(ref_box[1] * H), receptive_filed),
                       min(int(W - ref_box[2] * W), receptive_filed), min(int(H - ref_box[3] * H), receptive_filed)]
        _ref_box = [int(ref_box[0] * W)-ref_padding[0], int(ref_box[1] * H)-ref_padding[1],
                    int(ref_box[2] * W)+ref_padding[2], int(ref_box[3] * H)+ref_padding[3]]
        _tar_box = [int(tar_box[0] * W)-tar_padding[0], int(tar_box[1] * H)-tar_padding[1],
                    int(tar_box[2] * W)+tar_padding[2], int(tar_box[3] * H)+tar_padding[3]]

        ref, tar = crop_bbox_img(ref_img, _ref_box), crop_bbox_img(tar_img, _tar_box)

    if max(ref.shape) > max_unit_size:
        ref = downsample_img(ref, )
        tar = downsample_img(tar, )
        ref_padding = [int(ele / 2) for ele in ref_padding]
        tar_padding = [int(ele / 2) for ele in tar_padding]
    else:
        ref_padding = [int(ele) for ele in ref_padding]
        tar_padding = [int(ele) for ele in tar_padding]

    return [ref, tar], ref_padding, tar_padding


def padding_image_to_same(imgs, boxes_pd, swap=(2, 0, 1), dst_size=None):
    '''
    :param imgs: a list of imgs with different size
    :return: list of img with the max size of given imgs, constant padding. list of original img size
    '''
    if dst_size is None:
        dst_size = [200, 200]
    else:
        dst_size = list(dst_size)

    padded_img_list = []
    orininal_box_list = []
    for img, box_pd in zip(imgs, boxes_pd):
        H, W = img.shape[:2]
        dst_size[0], dst_size[1] = max(dst_size[0], H), max(dst_size[1], W)
        orininal_box_list.append([box_pd[0], box_pd[1], W - box_pd[2], H - box_pd[3]])
    for img in imgs:
        H, W = img.shape[:2]
        ph, pw = dst_size[0] - H, dst_size[1] - W
        padded_img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        # draw_and_show_boxes([], padded_img, )
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        padded_img_list.append(padded_img)
    return padded_img_list, orininal_box_list


def bbox_norm2abs(bbox, img_shape):
    """Conovert normalized bbox corrdinate to absolute pixel coordinate.

    Args:
        bbox (list | tuple | np.ndarray): Normalized bbox.
        img_shape (tuple | list): Image shape.

    Returns:
        Bbox in absolute pixel coordinate.
    """

    assert isinstance(bbox, (list, tuple, np.ndarray)), "Bbox should be list or tuple or np.ndarray!"
    bbox = np.array(bbox, dtype=np.float32)

    assert bbox.ndim <= 2
    if bbox.ndim == 1:
        assert bbox.size >= 4
    elif bbox.ndim == 2:
        assert bbox.shape[1] >= 4
    img_shape = np.array(img_shape)  # w, h
    bbox *= np.tile(img_shape, 2)
    bbox = np.array(bbox, dtype=int)
    return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]


# for saving files
import os


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # mp


import pickle


def save_pkl(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(name):
    return pickle.load(open(name, 'rb'))


def load_json(name):
    return json.load(open(name, 'r'))


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
