#!/usr/bin/env python
# coding=utf-8

#  General processing function
# including:
# singleton
# image
# tf model loading
# file or folder scans.
##########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from time import time
import numpy as np
import re
import cv2
from functools import wraps

import tensorflow as tf
from tensorflow.python.platform import gfile


def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[ class_ ] = class_(*args, **kwargs)
        return instances[ class_ ]

    return get_instance


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        t1 = time()
        # logger.debug("time running {}: {:.2f}ms".format(
        #     function.__name__, (t1 - t0) * 1000)
        # )
        print("time running {}: {:.2f}ms".format(
            function.__name__, (t1 - t0) * 1000)
        )
        return result

    return function_timer


def load_model(model, sess=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(sess, os.path.join(model_exp, ckpt_file))


def get_model_filenames_slim(model_dir):
    files = os.listdir(model_dir)
    meta_files = [ s for s in files if s.endswith('.meta') ]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1 and False:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[ -1 ]
    meta_files = [ s for s in files if '.ckpt' in s ]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[ 1 ])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[ 0 ]
    model_path = os.path.join(model_dir, ckpt_file)
    return model_path


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [ s for s in files if s.endswith('.meta') ]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1 and False:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[ 0 ]
    meta_files = [ s for s in files if '.ckpt' in s ]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[ 1 ])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[ 0 ]
    return meta_file, ckpt_file


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def annotate_points(im, points, verbose=0,color=(0, 0, 255)):
    # import ipdb; ipdb.set_trace()
    points = np.array(points)
    points = points.copy()
    points = points.astype(int)
    im = im.copy()
    for idx, point in enumerate(points):
        pos = (point[ 0 ], point[ 1 ])
        # print(pos)
        if verbose > 0:
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        # color=(0, 0, 255))
                        color=color)
        cv2.circle(im, pos, 1, color=color)
    return im


def _scan_image_tree(dir_path, img_list):
    for i, name in enumerate(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, name)
        if os.path.isdir(full_path):
            _scan_image_tree(full_path, img_list)
        else:
            img = full_path
            if (img.lower().endswith('.jpg') or img.lower().endswith('.png')
                or img.lower().endswith('.gif')
                or img.lower().endswith('.jpeg')
                or img.lower().endswith('.jpeg2000')
                or img.lower().endswith('.tif')
                or img.lower().endswith('.psg')
                or img.lower().endswith('.swf')
                or img.lower().endswith('.svg')
                or img.lower().endswith('.bmp')):
                img_list += [ img ]

            if len(img_list) % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write('\r #img of scan: %d' % (len(img_list)))


def scan_image_tree(dir_path):
    img_list = [ ]
    _scan_image_tree(dir_path, img_list)
    sys.stdout.write('\n')
    return img_list


def scan_image_dataset(dir_path, train_test_split=False, min_num_imgs=5, use_full_path=False, total_person_num=-1):
    id_list = [ ]
    img_list = [ ]
    cnts = [ ]
    names = [ ]
    if train_test_split:
        mask_train = [ ]
        cnts_tr = [ ]
    cnt_id = 0
    name_list = os.listdir(dir_path)
    name_list.sort()
    for i, name in enumerate(name_list):
        if i % 100 == 0:
            sys.stdout.flush()
            sys.stdout.write('\r #person of scan: %d' % (i))
        person_path = os.path.join(dir_path, name)
        if not os.path.isdir(person_path):
            continue
        ids = [ ]
        imgs = [ ]
        img_path_list = os.listdir(person_path)
        for img in img_path_list:
            if os.path.isdir(img):
                continue
            full_path_img = os.path.join(person_path, img)

            if (img.lower().endswith('.jpg') or img.lower().endswith('.png')
                or img.lower().endswith('.gif')
                or img.lower().endswith('.jpeg')
                or img.lower().endswith('.jpeg2000')
                or img.lower().endswith('.tif')
                or img.lower().endswith('.psg')
                or img.lower().endswith('.swf')
                or img.lower().endswith('.svg')
                or img.lower().endswith('.bmp')):
                ids += [ cnt_id ]
                if use_full_path:
                    imgs += [ full_path_img ]
                else:
                    imgs += [ name + '/' + img ]
                s = name + '/' + img
        id_list += ids
        img_list += imgs
        if len(ids) > min_num_imgs:
            cnt_id += 1
            if train_test_split:
                idx = np.random.permutation(len(ids))
                mask_train += list(idx != 0)
                cnts_tr += [ len(idx) - 1 ]
        cnts += [ len(ids) ]
        if total_person_num != -1 and len(cnts) > total_person_num:
            break
    sys.stdout.write('\n')
    if train_test_split:
        return img_list, id_list, cnts, mask_train, cnts_tr
    else:
        return img_list, id_list, cnts


def annotate_bbox(im, bbox):
    bbox = np.array(bbox)
    bbox = np.array(bbox).copy()
    bbox = bbox.astype(int)
    im2 = im.copy()
    # cv2.rectangle(im2, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)
    draw_rectangle(im2, bbox)
    return im2


def cvt_pts_to_shape(img, points):
    x1 = np.min(points[ :, 1 ])
    y1 = np.min(points[ :, 0 ])
    x2 = np.max(points[ :, 1 ])
    y2 = np.max(points[ :, 0 ])
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[ 1 ])
    y2 = min(y2, img.shape[ 0 ])
    bbox = [ x1, y1, x2 - x1, y2 - y1 ]
    return np.array(bbox)


def draw_rectangle(bgr_img, bbox):
    color = (0, 255, 255)
    # cv2.rectangle(bgr_img, (bbox[0], bbox[1]),
    #               (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 1)
    w = int(bbox[ 2 ] * 0.25)
    # h = int(bbox[3]  * 0.25)
    h = w
    x1 = bbox[ 0 ]
    y1 = bbox[ 1 ]
    # x2 = x1 + bbox[2];
    # y2 = y1 + bbox[3];
    x2 = bbox[ 2 ]
    y2 = bbox[ 3 ]
    thickness = 2
    # left top
    cv2.line(bgr_img, (x1, y1), (x1 + w, y1), color, thickness)
    cv2.line(bgr_img, (x1, y1), (x1, y1 + h), color, thickness)
    # left bottom
    cv2.line(bgr_img, (x1, y2), (x1 + w, y2), color, thickness)
    cv2.line(bgr_img, (x1, y2), (x1, y2 - h), color, thickness)
    # right bottom
    cv2.line(bgr_img, (x2, y2), (x2 - w, y2), color, thickness)
    cv2.line(bgr_img, (x2, y2), (x2, y2 - h), color, thickness)
    # right top
    cv2.line(bgr_img, (x2, y1), (x2 - w, y1), color, thickness)
    cv2.line(bgr_img, (x2, y1), (x2, y1 + h), color, thickness)


def annotate_shapes(im, shapes, verbose=0):
    # import ipdb; ipdb.set_trace()
    shapes = np.array(shapes)
    shapes = shapes.copy()
    shapes = shapes.astype(int)
    im = im.copy()
    for idx, point in enumerate(shapes):
        pos = (point[ 1 ], point[ 0 ])
        if verbose > 0:
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
        cv2.circle(im, pos, 2, color=(0, 255, 255))
    return im


def resize_bbox(bbox, img_w, img_h, scale=1.2):
    center_x = bbox[ 0 ] + bbox[ 2 ] / 2.0
    center_y = bbox[ 1 ] + bbox[ 3 ] / 2.0
    wh = max(bbox[ 2 ], bbox[ 3 ]) / 2.
    wh *= scale
    x1 = int(max(center_x - wh, 0))
    y1 = int(max(center_y - wh, 0))
    x2 = int(min(center_x + wh, img_w))
    y2 = int(min(center_y + wh, img_h))
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    bbox_ = [ x, y, w, h ]
    return bbox_


def cut_image_by_bbox(img, bbox, width, ext_width_scale=1.8, return_bbox=False, pts=None):
    center_x = bbox[ 0 ] + bbox[ 2 ] / 2.0
    center_y = bbox[ 1 ] + bbox[ 3 ] / 2.0
    wh = max(bbox[ 2 ], bbox[ 3 ]) / 2.
    wh *= ext_width_scale
    x1 = int(max(center_x - wh, 0))
    y1 = int(max(center_y - wh, 0))
    x2 = int(min(center_x + wh, img.shape[ 1 ]))
    y2 = int(min(center_y + wh, img.shape[ 0 ]))
    im_cut = img[ y1:y2, x1:x2 ]
    # print(bbox, x1, x2, y1, y2)
    im_cut = cv2.resize(im_cut, (width, width))
    s = width * 1. / (wh * 2)
    x = int((bbox[ 0 ] - x1) * s)
    y = int((bbox[ 1 ] - y1) * s)
    w = int(bbox[ 2 ] * s)
    h = int(bbox[ 3 ] * s)
    bbox_ = [ x, y, w, h ]
    if len(bbox) > 4:
        bbox_ += bbox[ 4: ]
    if pts is not None:
        pts[ :, 0 ] -= y1
        pts[ :, 1 ] -= x1
        pts[ :, 0 ] *= width / (y2 - y1)
        pts[ :, 1 ] *= width / (x2 - x1)
        return im_cut, pts
    if return_bbox:
        return im_cut, bbox_
    else:
        return im_cut


# def cut_image_by_bbox(img, bbox, width, ext_width_scale=1.8, return_bbox=False):
#     center_x = bbox[ 0 ] + bbox[ 2 ] / 2.0
#     center_y = bbox[ 1 ] + bbox[ 3 ] / 2.0
#     wh = max(bbox[ 2 ], bbox[ 3 ]) / 2.
#     wh *= ext_width_scale
#     x1 = int(max(center_x - wh, 0))
#     y1 = int(max(center_y - wh, 0))
#     x2 = int(min(center_x + wh, img.shape[ 1 ]))
#     y2 = int(min(center_y + wh, img.shape[ 0 ]))
#     im_cut = img[ y1:y2, x1:x2 ]
#     # print(bbox, x1, x2, y1, y2)
#     im_cut = cv2.resize(im_cut, (width, width))
#     s = width * 1. / (wh * 2)
#     x = int((bbox[ 0 ] - x1) * s)
#     y = int((bbox[ 1 ] - y1) * s)
#     w = int(bbox[ 2 ] * s)
#     h = int(bbox[ 3 ] * s)
#     bbox_ = [ x, y, w, h ]
#     if len(bbox) > 4:
#         bbox_ += bbox[ 4: ]
#     if return_bbox:
#         return im_cut, bbox_
#     else:
#         return im_cut


def cut_image_by_bboxes(img, bboxes, width, ext_width_scale=1.8):
    return np.array([ cut_image_by_bbox(img, bbox, width, ext_width_scale)
                      for bbox in bboxes ])


def cut_image(img, bbox):
    x1 = max(bbox[ 0 ], 0)
    y1 = max(bbox[ 1 ], 0)
    x2 = min(bbox[ 0 ] + bbox[ 2 ], img.shape[ 1 ])
    y2 = min(bbox[ 1 ] + bbox[ 3 ], img.shape[ 0 ])
    im_cut = img[ y1:y2, x1:x2 ]
    return im_cut


def cvt_68pts_to_5pts(points68):
    points5 = np.array([
        np.mean(points68[ 36:41 ], axis=0),  # left eye
        np.mean(points68[ 42:48 ], axis=0),  # right eye
        np.mean(points68[ 30:35 ], axis=0),  # node
        points68[ 48 ],  # left mouth
        points68[ 54 ] ]  # right mouth
    )
    return points5


def random_crop(value_list, size, seed=None, name=None):
    """Randomly crops a tensor to a given size.

    Slices a shape `size` portion out of `value` at a uniformly chosen offset.
    Requires `value.shape >= size`.

    If a dimension should not be cropped, pass the full size of that dimension.
    For example, RGB images can be cropped with
    `size = [crop_height, crop_width, 3]`.

    Args:
      value_list: Input tensor to crop.
      size: 1-D tensor with size the rank of `value`.
      seed: Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.
      name: A name for this operation (optional).

    Returns:
      A cropped tensor of the same rank as `value` and shape `size`.
    """
    # TODO(shlens): Implement edge case to guarantee output size dimensions.
    # If size > value.shape, zero pad the result so that it always has shape
    # exactly size.
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.ops import math_ops
    assert len(value_list) > 1
    with ops.name_scope(name, "random_crop", [ value_list[ 0 ], size ]) as name:
        value_list[ 0 ] = ops.convert_to_tensor(value_list[ 0 ], name="value")
        size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
        shape = array_ops.shape(value_list[ 0 ])
        check = control_flow_ops.Assert(
            math_ops.reduce_all(shape >= size),
            [ "Need value.shape >= size, got ", shape, size ],
            summarize=1000)
        shape = control_flow_ops.with_dependencies([ check ], shape)
        limit = shape - size + 1
        offset = tf.random_uniform(
            array_ops.shape(shape),
            dtype=size.dtype,
            maxval=size.dtype.max,
            seed=seed) % limit
        return [ array_ops.slice(v, offset, size, name=name) for v in value_list ]


def get_bbox_by_points(points):
    x_min = np.min(points[ :, 0 ])
    x_max = np.max(points[ :, 0 ])
    y_min = np.min(points[ :, 1 ])
    y_max = np.max(points[ :, 1 ])
    return [ x_min, y_min, x_max - x_min, y_max - y_min ]


def write_bbox(filename, bboxes, points):
    with open(filename, 'w') as f:
        s = ' '.join([ str(l) for l in bboxes[ 0 ][ :4 ] ])
        s += ' ' + ' '.join([ str(l) for l in points[ 0 ].ravel() ])
        f.write(s + '\n')


def read_bbox(filename):
    with open(filename) as f:
        for line in f.readlines():
            ll = line.strip().split(' ')
            if len(ll) > 0:
                name = ll[ 0 ]
                bbox = np.array([ int(float(l)) for l in ll[ :4 ] ])
                point = np.array([ float(l) for l in ll[ 4: ] ])
                point = point.reshape([ -1, 2 ])
                bboxes = np.array([ bbox ])
                points = np.array([ point ])
                return bboxes, points
        return None, None


def scale_image_larger(im):
    width = 500
    scale = width / im.shape[ 1 ]
    im_scale = cv2.resize(im, (int(im.shape[ 1 ] * scale),
                               int(im.shape[ 0 ] * scale)))
    return im_scale


def scan_image_tree_get_relative_path(root_path, selected_by=None, end_withs=None, is_save_relative=True):
    ''' 索引包含指定格式文件的地址或者相对地址
    :param root_path:
    :param relative_flag:
    :param selected_by: default None
    :param end_withs: list or str,default None
    :return:
    '''

    def _scan_image_tree(dir_path, pth_list, relative_path=None, is_save_relative=True):
        if relative_path is None:
            this_folder = dir_path
        else:
            this_folder = os.path.join(dir_path, relative_path)
        for i, name in enumerate(os.listdir(this_folder)):
            if relative_path is None:
                temp = name
            else:
                temp = os.path.join(relative_path, name)

            full_path = os.path.join(this_folder, name)
            if os.path.isdir(full_path):
                _scan_image_tree(dir_path, pth_list, temp, is_save_relative=is_save_relative)
            else:
                if is_save_relative:
                    pth = temp
                else:
                    pth = full_path

                if end_withs is None:
                    if selected_by is not None or selected_by not in pth:
                        # if select_by is None or select_by in path
                        pth_list += [ pth ]
                        # if is_save_relative:
                        #     pth_list += [ pth ]
                        # else:
                        #     pth_list += [ full_path ]
                elif isinstance(end_withs, list):
                    temp = [ True for e in end_withs if pth.lower().endswith(e) ]
                    if len(temp) > 0:
                        if selected_by is None or selected_by in pth:
                            pth_list += [ pth ]
                            # if is_save_relative:
                            #     pth_list += [ pth ]
                            # else:
                            #     pth_list += [ full_path ]

                if len(pth_list) % 100 == 0:
                    sys.stdout.flush()
                    sys.stdout.write('\r #img of scan: %d' % (len(pth_list)))

    list_output = [ ]
    _scan_image_tree(root_path, list_output, is_save_relative=is_save_relative)
    sys.stdout.write('\n')
    return list_output


def scan_folder_tree_get_relative_path(root_path, selected_by=None, end_withs=None, is_save_relative=True):
    '''  索引包含指定格式文件的上级文件夹地址或者相对地址
    :param root_path:
    :param relative_flag:
    :param selected_by: default None
    :param end_withs: list or str,default None
    :return:
    '''

    def _scan_folder_tree(dir_path, pth_list, relative_path=None, is_save_relative=True):
        if len(pth_list) % 10 == 0:
            sys.stdout.flush()
            sys.stdout.write('\r #img of scan: %d' % (len(pth_list)))

        if relative_path is None:
            this_folder = dir_path
        else:
            this_folder = os.path.join(dir_path, relative_path)

        for i, name in enumerate(os.listdir(this_folder)):
            if relative_path is None:
                temp = name
            else:
                temp = os.path.join(relative_path, name)

            full_path = os.path.join(this_folder, name)
            if os.path.isdir(full_path):
                _scan_folder_tree(dir_path, pth_list, temp, is_save_relative=is_save_relative)
            else:
                if is_save_relative:
                    pth = temp
                else:
                    pth = full_path

                if end_withs is None:
                    # all format file
                    if selected_by is not None or selected_by not in pth:
                        # if select_by is None or select_by in path
                        pth_list += [ os.path.dirname(pth) ]
                        return
                elif isinstance(end_withs, list):
                    # if contain the endswith format files
                    temp = [ True for e in end_withs if pth.lower().endswith(e) ]
                    if len(temp) > 0:
                        # contains the file
                        if selected_by is None or selected_by in pth:
                            # if select_by is None or select_by in path
                            pth_list += [ os.path.dirname(pth) ]
                            return

    list_output = [ ]
    _scan_folder_tree(root_path, list_output, is_save_relative=is_save_relative)
    sys.stdout.write('\n')
    return list_output
