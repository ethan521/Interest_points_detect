# coding=utf-8
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from theano.gof.cmodule import importlib
import common
import cv2
from face_attr.face_data_augment import *
# 设置模型参数
from common import scan_image_tree
from models import hourglass_arg_scope_tf
from tensorflow.python import pywrap_tensorflow
import pickle

learning_rate_init = 3e-5
learning_rate_decay = 0.9
training_epochs = 8000
num_batch_each_epoch = 50

batch_size = 200  # r36s2
batch_size = 800  # r12s2
batch_size = 300  # r20s2
# batch_size = 600
cls_batch_size = 60
# batch_size = 60
# cls_batch_size = 30
property_batch_size = batch_size - cls_batch_size

weight_decay = 1e-5
# opt_name = 'mom'
opt_name = 'adam'

# data augment
image_aug_proba = 0.1
random_rotate_angle = 20

# bottleneck_tensors_type = 'flatten'
bottleneck_tensors_type = 'avgpool'
use_signle_bottleneck = True

use_sigma_loss_weight = False
use_sigma_nn = False

loss_scale = 0.1
loss_weight = {
    'blur': loss_scale,
    'pts68': loss_scale * 0.1,
    'angle': loss_scale * 0.5,
    'age': loss_scale * 0.1,
    'beauty': loss_scale * 0.1,
    'expression': 1.8,
    'race': loss_scale * 0.1,
    'pts4': loss_scale * 0.5,
    'glasses': loss_scale * 0.1,
    'gender': 0.1,
    'occlusion': loss_scale * 0.1,
    'occleye': loss_scale * 0.1,
    'occreye': loss_scale * 0.1,
    'occmouth': loss_scale * 0.1,
    'cls': loss_scale * 0.1,
}

use_grayscale_image = False

# model_def = 'nets.mobilenet_v2'
# model_def = 'nets.irv1_stem_flatten'
# model_def = 'nets.irv1_small3'
# model_def = 'nets.irv1_stem_concat'
# model_def = 'nets.irv1_stem'
# model_def = 'nets.resface36s2_relu_avg'
# model_def = 'nets.resface36s2_relu'
# model_def = 'nets.resface12_relu_avg'
# model_def = 'nets.resface20s2_relu_avg'
model_def = 'nets.resface12s4_relu_avg'

# image_size = 96
# image_size = 48
# image_size = 64
# image_size = 160
image_size = 128

# pretrained_model_path = r"running_tmp\20190307-1630-pts68-resface20_relu_avg-96x96-expr8-attr"
# finetuning
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200115-2230-resface20_relu_avg-96x96-attr-rotate10-uncertain-pts-age"
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200115-2240-resface20_relu_avg-96x96-attr-rotate10-uncertain-pts-age-aug0.8"
pretrained_model_path = r""
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200115-1510-resface12s4_relu_avg-96x96-attr-rotate10-uncertain-pts-age-aug0.8"
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200117-0210-resface12s4_relu_avg-96x96-attr-rotate10-uncertain-pts-age-aug0.8"
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200316-1600-resface12s4_relu_flatten-96x96-attr-rotate10-uncertain-pts-age-aug0.01"
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200316-2130-resface12s4_relu_flatten-96x96-attr-rotate10-aug0.01-expr0.88"
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200117-1600-resface12s4_relu_avg-96x96-attr-rotate10-uncertain-pts-age-aug0.8"
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200330-1800-resface12s4_relu_avg-128x128-attr-rotate20-aug0.1"
pretrained_model_path = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200330-1830-resface12s4_relu_avg-128x128-attr-rotate20-aug0.1"

train_dir = r"E:\chenkai\Face_Detection_Alignment-djk\face_attr\running_tmp\20200331-0950-resface12s4_relu_avg-128x128-attr-rotate20-aug0.1"
# pretrained_model_path = train_dir


class MultiLossLayer():
  def __init__(self, loss_list):
    self._loss_list = loss_list
    self._sigmas_sq = []
    for i in range(len(self._loss_list)):
      self._sigmas_sq.append(slim.variable('Sigma_sq_' + str(i), dtype=tf.float32, shape=[], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))

  def get_loss(self):
    loss = 0
    for i in range(0, len(self._sigmas_sq)):
      factor = tf.div(1.0, tf.multiply(2.0, tf.maximum(1e-6, self._sigmas_sq[i])))
      loss += tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i]))
    return loss


def calc_multi_loss(loss_list):
  multi_loss_layer = MultiLossLayer(loss_list)
  return multi_loss_layer.get_loss()


def get_model_filenames_slim(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1 and False:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[-1]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        import re
        step_str = re.match(r'(^model.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    model_path = os.path.join(model_dir, ckpt_file)
    return model_path


def eval_tpr_far(y, prob, filename_output):
    y = 1 - np.array(y)
    prob = np.array(prob)
    print(filename_output)
    with open(filename_output, 'a') as f:
        cnt_blur = 0.
        cnt_clear = 0.
        idx = np.argsort(prob)
        total_n = np.sum(y == 1)
        total_p = np.sum(y == 0)
        s = 'total_p %d total_n %d' % (total_p, total_n)
        print(s)
        f.write(s + '\n')
        temp = -1
        for cnt, i in enumerate(idx):
            if y[i] == 1:
                cnt_clear += 1
            else:
                cnt_blur += 1
            tpr = cnt_clear / total_n
            far = cnt_blur / total_p
            disp_far_list = [1 / total_p, 0.0001, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            if cnt_blur > temp:
                for disp_far in disp_far_list:
                    if cnt_blur == int(disp_far * total_p):
                        temp = cnt_blur
                        s = 'cnt_clear: %5d  cnt_blur: %3d  label: %d  blur: %f  clean: %f  tpr: %f  far: %f' % (
                            cnt_clear, cnt_blur, int(y[i]), prob[i], 1 - prob[i], tpr, disp_far)
                        print(s)
                        f.write(s + '\n')
                        break


def eval_tpr_far_cls(y, prob, filename_output, s_name_pos='face', s_name_neg='noface'):
    '''
    :param y:  0 is noface, 1 is face
    :param prob:
    :param filename_output:
    :return:
    '''
    # y = 1 - np.array(y)
    y = np.array(y)
    prob = np.array(prob)
    print(filename_output)
    with open(filename_output, 'a') as f:
        cnt_p = 0.
        cnt_n = 0.
        idx = np.argsort(-prob)
        total_n = np.sum(y != 1)
        total_p = np.sum(y == 1)
        s = 'total_p %d total_n %d' % (total_p, total_n)
        print(s)
        f.write(s + '\n')
        temp = -1
        for cnt, i in enumerate(idx):
            if y[i] == 1:
                cnt_p += 1
            else:
                cnt_n += 1
            # tpr = cnt_n / total_n
            tpr = cnt_p / total_p
            # far = cnt_n / total_n
            # far = cnt_p / total_p

            # disp_far_list = [1 / total_n, 0.0001, 0.0003, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
            disp_far_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
            if cnt_n > temp:
                for disp_far in disp_far_list:
                    if cnt_n == int(disp_far * total_n):
                        temp = cnt_n
                        # tpr: 通过率
                        s = 'cnt_p: %5d  cnt_n: %5d  label: %d  %s: %f  %s: %f  tpr: %f  far: %f' % (
                            cnt_p, cnt_n, int(y[i]), s_name_pos, prob[i], s_name_neg, 1 - prob[i], tpr, disp_far)
                        print(s)
                        f.write(s + '\n')
                        break


def save_variables_and_metagraph(sess, saver, model_dir, step):
    # Save the model checkpoint
    import time
    print('Save variables ' + model_dir)
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model.meta')
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Save metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)


def get_data_label(file_path_list):
    blur_list = []
    pts68_list = []
    idx_list = []
    angle_list = []
    age_list = []
    beauty_list = []
    expression_list = []
    race_list = []
    pts4_list = []
    glasses_list = []
    gender_list = []
    occleye_list = []
    occreye_list = []
    occmouth_list = []
    for i, path in enumerate(file_path_list):
        # path = r'F:\data\face-attr\zhangbiao\96\positive-with-attr-label\img_00074996.jpg'
        sys.stdout.flush()
        sys.stdout.write('\r %d / %d' % (i, len(file_path_list)))
        path_face_attr = path[:-4] + '.detect.json'
        if not os.path.exists(path_face_attr):
            continue
        import json
        with open(path_face_attr) as f:
            face_attr_dict = json.load(f)
            if 'result_num' not in face_attr_dict or face_attr_dict['result_num'] == 0:
                continue
            result = face_attr_dict['result'][0]
            blur = result['qualities']['blur']
            pitch = result['pitch']
            yaw = result['yaw']
            roll = result['roll']
            occlusion_result = result['qualities']['occlusion']
            occ_labels = [occlusion_result['left_eye'], occlusion_result['right_eye'], occlusion_result['mouth']]
        pts68 = load_pts_label(path)
        if pts68 is None:
            continue
        pts68_list += [pts68.ravel() / image_size]
        blur_list += [blur]
        idx_list += [i]
        angle_list += [[pitch, yaw, roll]]
        age_list += [result['age']]
        beauty_list += [result['beauty']]
        expression_list += [int(result['expression'])]
        race_to_int = {'yellow': 0, 'white': 1, 'black': 2, 'arabs': 3}
        race_list += [race_to_int[result['race']]]
        pts4_list += [pts68.ravel() / image_size]
        glasses_list += [int(result['glasses'])]
        gender_to_int = {'male': 0, 'female': 1}
        gender_list += [gender_to_int[result['gender']]]
        occleye_list += [occ_labels[0]]
        occreye_list += [occ_labels[1]]
        occmouth_list += [occ_labels[2]]
        if False:
            occ_total = np.sum(occ_labels)
            if occ_labels[-1] > -0.3:
                print('occ_labels_ret', occ_labels)
                img = cv2.imread(path)
                bbox = common.cvt_pts_to_shape(img, pts68)
                print(bbox)
                img = common.annotate_bbox(img, bbox)
                img = common.annotate_shapes(img, pts68)
                cv2.imshow('img', img)
                key = cv2.waitKey(0)
                if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
                    exit(0)
                continue
    sys.stdout.write('\n')
    labels_dict = {}
    labels_dict['blur'] = np.array(blur_list)
    labels_dict['pts68'] = np.array(pts68_list)
    labels_dict['angle'] = np.array(angle_list)
    labels_dict['age'] = np.array(age_list)
    labels_dict['beauty'] = np.array(beauty_list)
    labels_dict['expression'] = np.array(expression_list)
    labels_dict['race'] = np.array(race_list)
    labels_dict['pts4'] = np.array(pts4_list)
    labels_dict['glasses'] = np.array(glasses_list)
    labels_dict['gender'] = np.array(gender_list)
    labels_dict['occleye'] = np.array(occleye_list)
    labels_dict['occreye'] = np.array(occreye_list)
    labels_dict['occmouth'] = np.array(occmouth_list)
    file_list = file_path_list[idx_list]
    labels_dict = transform_labels_list(labels_dict, verbose=1)
    return file_list, labels_dict


def get_data_image_and_label(file_path_list, image_size, is_training=False):
    img_list = []
    blur_list = []
    pts68_list = []
    angle_list = []
    age_list = []
    beauty_list = []
    expression_list = []
    race_list = []
    pts4_list = []
    glasses_list = []
    gender_list = []
    occlusion_list = []
    occleye_list = []
    occreye_list = []
    occmouth_list = []
    for path in file_path_list:
        path_face_attr = path[:-4] + '.detect.json'
        if not os.path.exists(path_face_attr):
            print('path', path)
            print('path', path.strip())
            print('path', path[:-4])
            print('path', path[-1])
            print('get_data_image_and_label no file', path_face_attr)
            exit(0)
        pts68 = load_pts_label(path)
        assert pts68 is not None
        import json
        with open(path_face_attr) as f:
            face_attr_dict = json.load(f)
            if 'result_num' not in face_attr_dict or face_attr_dict['result_num'] == 0:
                print('result_num error--', path_face_attr)
                exit(0)
            result = face_attr_dict['result'][0]
            blur = result['qualities']['blur']

            pitch = result['pitch']
            yaw = result['yaw']
            roll = result['roll']

            bbox = [0] * 4
            bbox[0] = int(result['location']['left'])
            bbox[1] = int(result['location']['top'])
            bbox[2] = int(result['location']['width'])
            bbox[3] = int(result['location']['height'])
            pts4 = []
            for point in result['landmark']:
                pts4 += [[point['y'], point['x']]]
            img = cv2.imread(path)
            if img is None or img.shape[0] <= 0:
                import menpo.io as mio
                image = mio.import_image(path)
                imgRgb = image.pixels_with_channels_at_back() * 255
                img = cv2.cvtColor(imgRgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if img is None or img.shape[0] <= 0:
                print(path)
                exit(0)
            if np.random.random() < 0.8:
                bbox = common.cvt_pts_to_shape(img, pts68)
            if is_training:
                bbox = random_bbox(img, bbox, hw_vari=0.25)
                ext_width_scale = 0.9 + np.random.random() * 0.5
            else:
                bbox = random_bbox(img, bbox, hw_vari=0.1)
                ext_width_scale = 1.2
            pts_all = np.concatenate([pts68, pts4], axis=0)
            img, pts_all = common.cut_image_by_bbox(
                img, bbox, width=image_size, ext_width_scale=ext_width_scale, pts=pts_all, swithxy=False)
            if img is None:
                print(path)
                bbox = common.cvt_pts_to_shape(img, pts68)
                print(bbox)
            pts68 = pts_all[:len(pts68)]
            pts4 = pts_all[len(pts68):]

            occlusion_result = result['qualities']['occlusion']
            occ_labels_ret = [occlusion_result['left_eye'], occlusion_result['right_eye'], occlusion_result['mouth']]
            occ_total = int(np.sum(occ_labels_ret) > 0.9)
            img, occ_labels = random_occlusion_keypoints(img, pts68, occ_proba=0.01)
            for i_occ in range(len(occ_labels)):
                if occ_labels_ret[i_occ] > 0.7:
                    occ_labels[i_occ] = 1
            occ_total += np.sum(occ_labels)
            occ_total = int(occ_total > 0.9)

            if is_training or True:
                # random rotate
                # angle = 80 - np.random.random() * 80 * 2
                # angle = 170 - np.random.random() * 170 * 2
                # angle = 45 - np.random.random() * 90
                # angle = 20 - np.random.random() * 20 * 2
                # angle = 10 - np.random.random() * 10 * 2
                # random_rotate_angle = 10
                angle = random_rotate_angle - np.random.random() * random_rotate_angle * 2
                img = rotate_image(img, -angle, False)
                pts68 = rotate_points(pts68, img.shape[1]//2, img.shape[0]//2, angle)
                pts4 = rotate_points(pts4, img.shape[1]//2, img.shape[0]//2, angle)
                bbox = common.cvt_pts_to_shape(img, pts68)
                roll += angle
                roll = roll % 360
                if roll > 180:
                    roll -= 360

            if img is None or img.shape[0] <= 0:
                print(path)
                exit(0)
            if is_training:
                img = image_augment_cv(img, aug_proba=image_aug_proba, isRgbImage=False, isNormImage=False)
            if use_grayscale_image:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if False:
                img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4))
                img = common.annotate_bbox(img, bbox * 4)
                img = common.annotate_shapes(img, pts68 * 4)
                img = common.annotate_shapes(img, pts4 * 4, color=(255, 0, 0))
                s = 'blur ' + str(blur)
                s += ' pitch %.0f yaw %.0f roll %.0f' % (pitch, yaw, roll)
                s += 'occ ' + str(occ_labels)
                print(s)
                cv2.putText(img, s, (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7,
                            color=(0, 0, 255))
                cv2.imshow('img', img)
                key = cv2.waitKey(0)
                if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
                    exit(0)
                continue
            # img = (img - 128.)/128.
            img = img / 256.
            img_list += [img]
            blur_list += [blur]
            pts68_list += [pts68.ravel() / image_size]
            angle_list += [[pitch, yaw, roll]]
            age_list += [result['age']]
            beauty_list += [result['beauty']]
            expression_list += [int(result['expression'])]
            race_to_int = {'yellow': 0, 'white': 1, 'black': 2, 'arabs': 3}
            race_list += [race_to_int[result['race']]]
            pts4_list += [pts4.ravel() / image_size]
            glasses_list += [int(result['glasses'])]
            gender_to_int = {'male': 0, 'female': 1}
            gender_list += [gender_to_int[result['gender']]]
            occlusion_list += [occ_total]
            occleye_list += [occ_labels[0]]
            occreye_list += [occ_labels[1]]
            occmouth_list += [occ_labels[2]]
    labels_dict = {}
    labels_dict['blur'] = np.array(blur_list)
    labels_dict['pts68'] = np.array(pts68_list)
    labels_dict['angle'] = np.array(angle_list)
    labels_dict['age'] = np.array(age_list)
    labels_dict['beauty'] = np.array(beauty_list)
    labels_dict['expression'] = np.array(expression_list)
    labels_dict['race'] = np.array(race_list)
    labels_dict['pts4'] = np.array(pts4_list)
    labels_dict['glasses'] = np.array(glasses_list)
    labels_dict['gender'] = np.array(gender_list)
    labels_dict['occlusion'] = np.array(occlusion_list)
    labels_dict['occleye'] = np.array(occleye_list)
    labels_dict['occreye'] = np.array(occreye_list)
    labels_dict['occmouth'] = np.array(occmouth_list)
    labels_dict = transform_labels_list(labels_dict)
    return np.array(img_list), labels_dict


def get_cls_image_and_label(file_path_list, image_size, aug_proba=1.0, is_training=False):
    img_list = []
    label_path = os.path.dirname(file_path_list[0])
    if 'positive' in label_path:
        label = 1
    elif 'part' in label_path:
        label = 0
    elif 'negative' in label_path:
        label = 0
    else:
        return [], []

    for path in file_path_list:
        img = cv2.imread(path)
        if img is None:
            continue
        if (img.shape[0] != image_size) or (img.shape[1] != image_size):
            img = cv2.resize(img, (image_size, image_size))
        if np.random.random() < aug_proba * 0.01:
            img = detail_enhance(img, np.random.random() * 4)
        if np.random.random() < aug_proba * 0.003:
            img = edge_preserve(img, np.random.random() * 3)
        if np.random.random() < aug_proba * 0.01:
            img = random_occlusion(img)
        # 饱和度
        if np.random.random() < aug_proba * 0.2:
            img = change_saturation(img, -20 + np.random.random() * 40)
        # 亮度
        if np.random.random() < aug_proba:
            img = change_darker(img, -8 + np.random.random() * 16)
        if np.random.random() < aug_proba * 0.8:
            img = random_rotate(img, 10, True)
        if np.random.random() < aug_proba * 0.8:
            img = random_pad(img, 0.005, 0.15)
        if np.random.random() < aug_proba * 0.8:
            img = random_crop(img, 0.7, 0.5)
        if np.random.random() < aug_proba * 0.3 or use_grayscale_image:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if np.random.random() < aug_proba * 0.3:
            img = 255 - img
        if np.random.random() < aug_proba * 0.1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, (image_size, image_size))
        # img = (img - 128.)/128.
        img = img / 256.
        img_list += [img]

    cls_label_list = np.ones(len(img_list)) * label
    return np.array(img_list), cls_label_list


def get_cls_image_test(file_path_list, image_size, aug_proba=1.0, is_training=False):
    img_list = []
    label_list = []
    for path in file_path_list:
        img = cv2.imread(path)
        if img is None:
            continue
        label_path = os.path.dirname(path)
        if 'positive' in label_path:
            label = 1
        elif 'part' in label_path:
            label = 0
        elif 'negative' in label_path:
            label = 0
        else:
            label = 0
        label_list.append(label)

        if (img.shape[0] != image_size) or (img.shape[1] != image_size):
            img = cv2.resize(img, (image_size, image_size))
        if np.random.random() < aug_proba * 0.01:
            img = detail_enhance(img, np.random.random() * 4)
        if np.random.random() < aug_proba * 0.003:
            img = edge_preserve(img, np.random.random() * 3)
        if np.random.random() < aug_proba * 0.01:
            img = random_occlusion(img)
        # 饱和度
        if np.random.random() < aug_proba * 0.1:
            img = change_saturation(img, -20 + np.random.random() * 40)
        # 亮度
        if np.random.random() < aug_proba:
            img = change_darker(img, -8 + np.random.random() * 16)
        if np.random.random() < aug_proba * 0.3 or use_grayscale_image:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if np.random.random() < aug_proba * 0.3:
            img = 255 - img
        if np.random.random() < aug_proba * 0.1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (image_size, image_size))

        # img = (img - 128.)/128.
        img = img / 256.
        img_list += [img]
    return np.array(img_list), label_list


def load_pts_label(filename):
    filename_pts = os.path.splitext(filename)[0] + '.pts'
    if not os.path.exists(filename_pts):
        return None
    with open(filename_pts) as f:
        gt_pts = []
        lines = f.readlines()
        for line in lines[3:3 + 68]:
            l = line.strip().split(' ')
            gt_pts += [[float(l[1]), float(l[0])]]
        return np.array(gt_pts)


def transform_labels_list(labels_dict, verbose=0):
    blur_threshold = 0.2
    pos_idxes = np.where(labels_dict['blur'] > 0.8)[0]
    neg_idxes = np.where(labels_dict['blur'] <= blur_threshold)[0]
    labels_dict['blur_pos_idxes'] = pos_idxes
    labels_dict['blur_neg_idxes'] = neg_idxes
    if verbose > 0:
        print('blur', 'pos-neg', len(pos_idxes), len(neg_idxes))
    labels_dict['blur'] = (labels_dict['blur'] > blur_threshold).astype(int)
    # age
    cond = np.bitwise_or(labels_dict['age'] < 20, labels_dict['age'] > 50)
    pos_idxes = np.where(cond)[0]
    neg_idxes = np.where(np.bitwise_not(cond))[0]
    labels_dict['age_pos_idxes'] = pos_idxes
    labels_dict['age_neg_idxes'] = neg_idxes
    # gender
    pos_idxes = np.where(labels_dict['gender'] == 1)[0]
    neg_idxes = np.where(labels_dict['gender'] == 0)[0]
    labels_dict['gender_pos_idxes'] = pos_idxes
    labels_dict['gender_neg_idxes'] = neg_idxes
    # expression smile
    expr_idxes0 = np.where(labels_dict['expression'] == 0)[0]
    expr_idxes1 = np.where(labels_dict['expression'] == 1)[0]
    expr_idxes2 = np.where(labels_dict['expression'] == 2)[0]
    labels_dict['expression_idxes_0'] = expr_idxes0
    labels_dict['expression_idxes_1'] = expr_idxes1
    labels_dict['expression_idxes_2'] = expr_idxes2
    return labels_dict


def transform_file_list(train_file_list, test_file_list, verbose=0):
    train_file_list = np.array(train_file_list)
    test_file_list = np.array(test_file_list)

    train_file_list, train_labels_dict = get_data_label(train_file_list)
    # test_label_list, test_file_list = get_data_label(train_file_list)
    print('train_file_list', train_file_list.shape)
    test_file_list, test_labels_dict = get_data_label(test_file_list)
    print('test_file_list', test_file_list.shape)
    if verbose > 0 or True:
        def statistic_of_labels(labels, name):
            if isinstance(labels[0], int):
                import collections
                print(collections.Counter(labels))
            else:
                if not os.path.exists(os.path.join(train_dir, 'hist')):
                    os.makedirs(os.path.join(train_dir, 'hist'))
                from matplotlib import pyplot
                pyplot.figure()
                pyplot.hist(labels, 20, label=name)
                pyplot.xlabel('value')
                pyplot.ylabel(name)
                pyplot.title(name)
                pyplot.legend(loc='upper right')
                path = os.path.join(train_dir, 'hist', name + ".png")
                print(path)
                pyplot.savefig(path)
                if verbose > 1:
                    pyplot.show()

        stat_list = ['age', 'blur', 'expression', 'race', 'glasses', 'gender', 'beauty', 'angle',
                     'occleye', 'occreye', 'occmouth']
        for k in stat_list:
            statistic_of_labels(train_labels_dict[k], 'train_' + k)
        for k in stat_list:
            statistic_of_labels(test_labels_dict[k], 'test_' + k)

    return train_file_list, train_labels_dict, test_file_list, test_labels_dict


def disp_data(X_batch, y_batch, blur_predicts=None):
    keys = list(y_batch.keys())
    keys.sort()
    print(keys)
    for i, x in enumerate(X_batch):
        x = np.copy(x)
        x = x * 128 + 128
        x = x.astype(np.uint8)
        print(x.shape)
        x = cv2.resize(x, (600, 600))
        s = 'blur ' + str(y_batch['blur'][i])
        if blur_predicts is not None:
            s += ' ' + str(blur_predicts[i])
        print(s)
        s = ''
        for k in keys:
            if 'pts68' in k:
                x = common.annotate_shapes(x, y_batch[k][i].reshape((-1, 2)) * 600)
            elif 'pts4' in k:
                x = common.annotate_shapes(x, y_batch[k][i].reshape((-1, 2)) * 600, color=(255, 0, 0))
            elif 'idxes' not in k:
                s += k + ' ' + str(y_batch[k][i]) + ',  '
            elif 'occlusion' not in k:
                s += k + ' ' + str(y_batch[k][i]) + ',  '
        print(s)
        cv2.putText(x, s, (10, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255))
        cv2.imshow('img', x)
        key = cv2.waitKey(0)
        if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
            break


def build_opt(cost, opt_name, learning_rate, step, variables_to_train):
    # 优化
    print('------- opt_name ------- ', opt_name)
    if opt_name == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif opt_name == 'adam-param':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-1)
    elif opt_name == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1e-1)
    elif opt_name == 'mom':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    optimizer = opt.minimize(cost, global_step=step, var_list=variables_to_train)
    return optimizer


def build_backbone_network(model_def, inputs, is_training=True, n_landmarks=68):
    # image_batch = tf.identity(inputs, 'image_batch')
    # image_batch = tf.identity(image_batch, 'input')
    image_batch = inputs
    print('Building training graph')
    # model_def = 'nets.irv1_stem'
    print(model_def)
    network = importlib.import_module(model_def)
    print('weight_decay', weight_decay)
    print(network)
    # Build the inference graph
    prelogits, end_points = network.inference(image_batch, 1.0, phase_train=is_training, weight_decay=weight_decay)
    return end_points


def build_network(images_input, model_def=model_def, n_landmarks=68, is_training=False, bottleneck_type=bottleneck_tensors_type):
    with tf.variable_scope('net'):
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
            with slim.arg_scope(hourglass_arg_scope_tf(use_fc_bn=False)):
                end_points = build_backbone_network(model_def, images_input, is_training=is_training,
                                                    n_landmarks=n_landmarks)
                with tf.variable_scope('net', 'MultiScaleLayer'):
                    def get_tensors_from_endpoints(net_temp, bottleneck_type='avgpool'):
                        if bottleneck_type == 'avgpool':
                            net_temp = slim.avg_pool2d(net_temp, net_temp.get_shape()[1:3], padding='VALID',
                                                       scope='AvgPool')
                        # net_temp = slim.flatten(net_temp)
                        print(net_temp, net_temp.get_shape())
                        net_temp = tf.reshape(net_temp, [-1, net_temp.get_shape()[1]*net_temp.get_shape()[2]*net_temp.get_shape()[3]])
                        return net_temp
                    # nets = []
                    # net_names = ['Conv2d_1a_3x3', 'Conv2d_2b_3x3', 'Conv2d_4a_3x3', 'Conv2d_4b_3x3']
                    # net_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4']
                    # for name in net_names:
                    #     nets += [get_tensors_from_endpoints(end_points[name])]
                    # net = tf.concat(nets, 1)

                net_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4']
                with tf.variable_scope('Group1'):
                    net_conv = get_tensors_from_endpoints(end_points[net_names[-1]], bottleneck_type)
                    net_pts_btl = slim.fully_connected(net_conv, 128, scope='pts_bottleneck', reuse=False)
                    net = net_pts_btl
                    end_points['pts68_predictions'] = slim.fully_connected(net, n_landmarks * 2, activation_fn=None,
                                                                           scope='pts68_prediction', reuse=False)
                    end_points['pts4_predictions'] = slim.fully_connected(net, 4 * 2, activation_fn=None,
                                                                          scope='pts4_predictions', reuse=False)
                    end_points['angle_predictions'] = slim.fully_connected(net, 3, activation_fn=None,
                                                                           scope='pred_angle', reuse=False)
                    if use_sigma_nn:
                        with tf.variable_scope('Sigma'):
                            net = slim.fully_connected(net_conv, 128, scope='sigma', reuse=False)
                            end_points['pts4_sigma'] = tf.exp(slim.fully_connected(net, 4 * 2, activation_fn=None,
                                                                                   scope='pts4_sigma', reuse=False))
                with tf.variable_scope('Group2'):
                    if use_signle_bottleneck:
                        net = net_pts_btl
                    else:
                        net = get_tensors_from_endpoints(end_points[net_names[-1]], bottleneck_type)
                        net = slim.fully_connected(net, 128, scope='attr_bottleneck', reuse=False)
                    # net = tf.concat([net, end_points['bottleneck']], 1)
                    end_points['blur_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                          scope='blur_predictions', reuse=False)
                    end_points['cls_predictions'] = slim.fully_connected(net, num_outputs=2,
                                                                         activation_fn=tf.nn.softmax,
                                                                         scope='cls_predictions', reuse=False)
                    end_points['glasses_predictions'] = slim.fully_connected(net, 3, activation_fn=None,
                                                                             scope='glasses_predictions', reuse=False)
                    end_points['occlusion_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                               scope='occlusion_predictions', reuse=False)
                    end_points['occleye_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                             scope='occleye_predictions', reuse=False)
                    end_points['occreye_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                             scope='occreye_predictions', reuse=False)
                    end_points['occmouth_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                              scope='occmouth_predictions', reuse=False)
                with tf.variable_scope('Group3'):
                    if use_signle_bottleneck:
                        net = net_pts_btl
                    else:
                        net_conv = get_tensors_from_endpoints(end_points[net_names[-1]], bottleneck_type)
                        net = slim.fully_connected(net_conv, 128, scope='age_bottleneck', reuse=False)
                    end_points['expression_predictions'] = slim.fully_connected(net, 3, activation_fn=None,
                                                                                scope='expression_predictions',
                                                                                reuse=False)
                    end_points['beauty_predictions'] = slim.fully_connected(net, 1, activation_fn=None,
                                                                            scope='beauty_predictions', reuse=False)
                    end_points['race_predictions'] = slim.fully_connected(net, 4, activation_fn=None,
                                                                          scope='race_predictions', reuse=False)
                    end_points['age_predictions'] = slim.fully_connected(net, 1, activation_fn=None,
                                                                         scope='age_predictions', reuse=False)
                    # net = slim.fully_connected(net_conv, 128, scope='gender_bottleneck', reuse=False)
                    end_points['gender_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                            scope='gender_predictions', reuse=False)
                    if use_sigma_nn:
                        with tf.variable_scope('Sigma'):
                            net = slim.fully_connected(net_conv, 128, scope='sigma', reuse=False)
                            end_points['age_sigma'] = tf.exp(slim.fully_connected(net, 1, activation_fn=None,
                                                                              scope='age_sigma', reuse=False))
                # with tf.variable_scope('Group4'):
                #     net = get_tensors_by_avgpool(end_points[net_names[-1]])
                #     end_points['age_predictions'] = slim.fully_connected(net, 1, activation_fn=None,
                #                                                          scope='age_predictions', reuse=False)
                # with tf.variable_scope('Group5'):
                #     net = get_tensors_by_avgpool(end_points[net_names[-1]])
                #     net = slim.fully_connected(net, 128, scope='age_bottleneck', reuse=False)
                #     end_points['gender_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                #                                                             scope='gender_predictions', reuse=False)

                return end_points


def build_loss(end_points, label_points, attr_inds):
    loss_points = {}

    # blur cost
    blur_logits = tf.gather(end_points['blur_predictions'], attr_inds)
    y_input_one_hot = tf.one_hot(label_points['blur'], 2)
    # cost_blur = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=logits, labels=y_input))
    cost_blur = tf.reduce_mean(slim.losses.softmax_cross_entropy(
        logits=blur_logits, onehot_labels=y_input_one_hot, label_smoothing=0.1))
    # vhard = tf.contrib.distributions.percentile(cost_blur, 95)
    # cost_blur = tf.where(cost_blur>vhard, cost_blur*0.001, cost_blur)
    loss_points['blur'] = cost_blur

    # landmark cost
    m = 10
    pts68_pred = tf.gather(end_points['pts68_predictions'], attr_inds)
    # l2norm = slim.losses.mean_squared_error(predictions*m, gt_lms*m)
    cost_pts68 = slim.losses.absolute_difference(pts68_pred * m, label_points['pts68'] * m)
    loss_points['pts68'] = cost_pts68

    if 'angle' in label_points:
        angle_pre = tf.gather(end_points['angle_predictions'], attr_inds)
        loss_points['angle'] = 0.9 * slim.losses.absolute_difference(
            angle_pre / 10, label_points['angle'] / 10)
    if 'age' in label_points:
        age_pre = tf.gather(end_points['age_predictions'], attr_inds)
        loss_points['age'] = 10.0 * slim.losses.absolute_difference(
            tf.reshape(age_pre, (-1,)) / 100, label_points['age'] / 100)

        if 'age_sigma' in end_points:
            age_diff = tf.abs(tf.reshape(age_pre, (-1,)) / 100 - label_points['age'] / 100)
            age_sigma = tf.reshape(end_points['age_sigma'], (-1,))
            age_sigma = tf.gather(age_sigma, attr_inds)
            loss_points['age'] = 10 * tf.reduce_mean(age_diff / age_sigma + tf.log(age_sigma))

    if 'beauty' in label_points:
        beauty_pred = tf.reshape(tf.gather(end_points['beauty_predictions'], attr_inds), (-1,))
        loss_points['beauty'] = 10.0 * slim.losses.absolute_difference(
            beauty_pred / 100, label_points['beauty'] / 100)

    if 'expression' in label_points:
        expression_pre = tf.gather(end_points['expression_predictions'], attr_inds)
        loss_points['expression'] = tf.reduce_mean(
            slim.losses.sparse_softmax_cross_entropy(
                logits=expression_pre, labels=label_points['expression']))

    if 'race' in label_points:
        race_pre = tf.gather(end_points['race_predictions'], attr_inds)
        loss_points['race'] = tf.reduce_mean(
            slim.losses.sparse_softmax_cross_entropy(logits=race_pre, labels=label_points['race']))

    if 'pts4' in label_points:
        pts4_pre = tf.gather(end_points['pts4_predictions'], attr_inds)
        loss_points['pts4'] = slim.losses.absolute_difference(
            pts4_pre * 10, label_points['pts4'] * 10)
        if 'pts4_sigma' in end_points:
            pts4_diff = tf.abs(pts4_pre - label_points['pts4'])
            pts4_sigma = end_points['pts4_sigma']
            pts4_sigma = tf.gather(pts4_sigma, attr_inds)
            loss_points['pts4'] = 10 * tf.reduce_mean(tf.reduce_mean(pts4_diff / pts4_sigma + tf.log(pts4_sigma), axis=1))
    if 'glasses' in label_points:
        glasses_pre = tf.gather(end_points['glasses_predictions'], attr_inds)
        loss_points['glasses'] = tf.reduce_mean(
            slim.losses.sparse_softmax_cross_entropy(
                logits=glasses_pre, labels=label_points['glasses']))

    if 'gender' in label_points:
        gender_pre = tf.gather(end_points['gender_predictions'], attr_inds)
        loss_points['gender'] = tf.reduce_mean(slim.losses.softmax_cross_entropy(
            logits=gender_pre, onehot_labels=tf.one_hot(label_points['gender'], 2),
            label_smoothing=0.1))

    if 'occlusion' in label_points:
        occlusion_pre = tf.gather(end_points['occlusion_predictions'], attr_inds)
        loss_points['occlusion'] = tf.reduce_mean(slim.losses.softmax_cross_entropy(
            logits=occlusion_pre, onehot_labels=tf.one_hot(label_points['occlusion'], 2),
            label_smoothing=0.1))
    if 'occleye' in label_points:
        occleye_pre = tf.gather(end_points['occleye_predictions'], attr_inds)
        loss_points['occleye'] = tf.reduce_mean(slim.losses.softmax_cross_entropy(
            logits=occleye_pre, onehot_labels=tf.one_hot(label_points['occleye'], 2),
            label_smoothing=0.1))

    if 'occreye' in label_points:
        occreye_pre = tf.gather(end_points['occreye_predictions'], attr_inds)
        loss_points['occreye'] = tf.reduce_mean(slim.losses.softmax_cross_entropy(
            logits=occreye_pre, onehot_labels=tf.one_hot(label_points['occreye'], 2),
            label_smoothing=0.1))

    if 'occmouth' in label_points:
        occmouth_pre = tf.gather(end_points['occmouth_predictions'], attr_inds)
        loss_points['occmouth'] = tf.reduce_mean(slim.losses.softmax_cross_entropy(
            logits=occmouth_pre, onehot_labels=tf.one_hot(label_points['occmouth'], 2),
            label_smoothing=0.1))

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    assert len(regularization_losses) > 0
    regularization_losses = [tf.cast(l, tf.float32) for l in regularization_losses]
    regularizers = tf.add_n(regularization_losses, name='total_loss')
    loss_points['regu'] = regularizers
    return loss_points


def build_loss_face_cls(cls_prob, label):
    if False:
        label = tf.to_int32(label)
        loss = 1.0 * tf.reduce_mean(slim.losses.softmax_cross_entropy(
            logits=cls_prob, onehot_labels=tf.one_hot(label, 2), label_smoothing=0.1))
        return loss
    zeros = tf.zeros_like(label)
    # label=-1 --> label=0net_factory

    # pos -> 1, neg -> 0, others -> 0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)
    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    # row = [0,2,4.....]
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)

    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros, zeros, ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)
    num_keep_radio = 0.9
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def build_input_placeholder(batch_size, cls_batch_size):
    label_points = {}
    # label分为两部分，属性部分，
    property_batch_size = batch_size - cls_batch_size
    label_points['blur'] = tf.placeholder('int32', [property_batch_size, ])
    label_points['pts68'] = tf.placeholder('float', [property_batch_size, 68 * 2])
    label_points['angle'] = tf.placeholder('float', [property_batch_size, 3])
    label_points['pts4'] = tf.placeholder('float', [property_batch_size, 4 * 2])
    label_points['glasses'] = tf.placeholder('int32', [property_batch_size, ])
    label_points['age'] = tf.placeholder('float', [property_batch_size, ])
    # label_points['beauty'] = tf.placeholder('float', [property_batch_size, ])
    label_points['expression'] = tf.placeholder('int32', [property_batch_size, ])
    # label_points['race'] = tf.placeholder('int32', [property_batch_size, ])
    label_points['gender'] = tf.placeholder('int32', [property_batch_size, ])
    # label_points['occlusion'] = tf.placeholder('int32', [property_batch_size, ])
    # label_points['occleye'] = tf.placeholder('int32', [property_batch_size, ])
    # label_points['occreye'] = tf.placeholder('int32', [property_batch_size, ])
    # label_points['occmouth'] = tf.placeholder('int32', [property_batch_size, ])
    label_points['cls'] = tf.placeholder('float32', [cls_batch_size, ])
    label_points['cls'] = tf.placeholder('float32', [batch_size, ])
    return label_points

def get_acc_points_bk(label_points, end_points, attr_valid_inds):
    acc_list = ['blur', 'expression', 'race', 'glasses', 'gender']
    acc_points = {}
    pred_points = {}
    for name in acc_list:
        if name in label_points.keys():
            logits = tf.gather(end_points[name + '_predictions'], attr_valid_inds)
            predict_prob = tf.nn.softmax(logits)
            predicted = tf.cast(tf.arg_max(logits, 1), tf.int32)
            accuracy_clf = tf.reduce_mean(tf.cast(tf.equal(predicted, label_points[name]), tf.float32))
            acc_points[name] = accuracy_clf
            pred_points[name] = predicted

    pred_points['age'] = end_points['age_predictions']
    pred_points['pitch'] = tf.gather(end_points['angle_predictions'], 0, axis=1)
    pred_points['yaw'] = tf.gather(end_points['angle_predictions'], 1, axis=1)
    pred_points['roll'] = tf.gather(end_points['angle_predictions'], 2, axis=1)
    pred_points['beauty'] = end_points['beauty_predictions']
    pred_points['pts4'] = end_points['pts4_predictions']
    pred_points['pts68'] = end_points['pts68_predictions']
    return acc_points, pred_points

def get_acc_points(label_points, end_points, attr_valid_inds, cls_valid_inds):
    acc_list = ['blur', 'expression', 'race', 'glasses', 'gender', 'cls',
                'occlusion', 'occleye', 'occreye', 'occmouth']
    acc_points = {}
    pred_points = {}
    for name in acc_list:
        if name in label_points.keys():
            if name + '_predictions' not in end_points:
                continue
            if 'cls' in name:
                logits = tf.gather(end_points[name + '_predictions'], cls_valid_inds)
                # predicted = tf.cast(tf.arg_max(logits, 1), tf.int32)
                # accuracy_clf = tf.reduce_mean(tf.cast(tf.equal(predicted, label_points[name]), tf.float32))
                predicted = tf.cast(tf.arg_max(logits, 1), tf.float32)
                pass
            else:
                logits = tf.gather(end_points[name + '_predictions'], attr_valid_inds)
                # predict_prob = tf.nn.softmax(logits)
                predicted = tf.cast(tf.arg_max(logits, 1), tf.int32)
            accuracy_clf = tf.reduce_mean(tf.cast(tf.equal(predicted, label_points[name]), tf.float32))
            acc_points[name] = accuracy_clf
            pred_points[name] = predicted
    return acc_points

def get_test_image_list(root_folder):
    folder_list = os.listdir(root_folder)
    image_list = []
    for folder in folder_list:
        path = os.path.join(root_folder, folder)
        temp = [os.path.join(path, e) for e in os.listdir(path)]
        image_list += temp
    return image_list


def get_images(folder, image_path_list):
    list_temp = os.listdir(folder)
    if len(list_temp) > 0:
        path = os.path.join(folder, list_temp[0])
        if os.path.isfile(path):
            image_path_list += [os.path.join(folder, e) for e in list_temp]
        else:
            for temp in list_temp:
                next_folder = os.path.join(folder, temp)
                get_images(next_folder, image_path_list)


def main_train(training_file_list_dict, only_eval=False):
    train_file_list = training_file_list_dict['attr_train_file_list']
    train_labels_dict = training_file_list_dict['attr_train_labels_dict']
    test_file_list = training_file_list_dict['attr_test_file_list']
    test_labels_dict = training_file_list_dict['attr_test_labels_dict']
    cls_image_list_neg = training_file_list_dict['cls_image_list_neg']
    cls_image_list_part = training_file_list_dict['cls_image_list_part']
    cls_image_list_pos = training_file_list_dict['cls_image_list_pos']
    cls_image_list_test = training_file_list_dict['cls_image_list_test']
    # test_root_folder = r'E:\datasets\public_datasets\WIDER\96\test'
    # face_image_list_test = get_test_image_list(test_root_folder)
    if only_eval:
        train_file_list = test_file_list
        verbose = 0
    else:
        verbose = 1
    print(len(train_file_list))
    print(len(test_file_list))
    # train_file_list, train_file_list_pos, train_file_list_neg, test_file_list
    # RESAMPLE_FLAG = False

    n_sample = len(train_file_list)
    # n_class = y_train.shape[1]

    phase_train_placeholder = True
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    x_input = tf.placeholder('float', [batch_size, image_size, image_size, 3])

    attr_valid_inds = tf.range(0, property_batch_size)
    # cls_valid_inds = tf.range(property_batch_size, batch_size)
    cls_valid_inds = tf.range(0, batch_size)

    label_points = build_input_placeholder(batch_size=batch_size, cls_batch_size=cls_batch_size)
    end_points = build_network(x_input, is_training=phase_train_placeholder)
    acc_points = get_acc_points(label_points, end_points, attr_valid_inds, cls_valid_inds)

    # cls loss
    cls_predict_prob = tf.gather(end_points['cls_predictions'], cls_valid_inds)
    cls_label = label_points['cls']
    # cls_loss = build_loss_face_cls(end_points, label_points)

    # add mask face classify loss
    cls_loss = build_loss_face_cls(cls_predict_prob, cls_label)

    # blur pred
    label_blur = label_points['blur']
    logits_blur = tf.gather(end_points['blur_predictions'], attr_valid_inds)
    blur_predict_prob = tf.nn.softmax(logits_blur)
    blur_predicted = tf.cast(tf.arg_max(logits_blur, 1), tf.int32)
    blur_accuracy_clf = tf.reduce_mean(tf.cast(tf.equal(blur_predicted, label_blur), tf.float32))

    # attr loss
    loss_points = build_loss(end_points, label_points, attr_valid_inds)

    # occ prob
    logits_occ = tf.gather(end_points['occmouth_predictions'], attr_valid_inds)
    occ_predict_prob = tf.nn.softmax(logits_occ)

    loss_points['cls'] = cls_loss


    if use_sigma_loss_weight:
        total_loss = calc_multi_loss([v for k, v in loss_points.items() if k != 'regu'])
        total_loss += loss_points['regu']
    else:
        loss_weight['regu'] = 1.
        # total loss
        total_loss = 0
        for k, v in loss_points.items():
            total_loss += loss_weight[k] * v

    step = tf.Variable(0, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(
        learning_rate_init,  # Base learning rate.
        step * batch_size,  # Current index into the dataset.
        n_sample,  # Decay step.
        learning_rate_decay,  # Decay rate.
        staircase=True)

    trainable_scopes = ['net/blur_bottleneck', 'net/blur_predictions']
    trainable_scopes = ['net/Group1/Sigma', 'net/Group3/Sigma']
    trainable_scopes = ['net/Conv4', 'net/Group1', 'net/Group2', 'net/Group3', 'net/Group4']
    trainable_scopes = []
    if len(trainable_scopes) == 0:
        variables_to_train = tf.global_variables()
    else:
        variables_to_train = []
        for scope in trainable_scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train += variables
        print('-------- variables_to_train --------')
        print(variables_to_train)

    optimizer = build_opt(total_loss, opt_name, learning_rate, step, variables_to_train)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # 初始化所有变量
    init = tf.initialize_all_variables()

    variables_to_restore = tf.all_variables()
    variables_to_restore = tf.trainable_variables()
    variables_to_restore = slim.get_model_variables()
    saver = tf.train.Saver(variables_to_restore, max_to_keep=1)
    with open(os.path.join(train_dir, 'training-log.txt'), 'w') as f:
        print('training')

    acc_keys = list(acc_points.keys())
    loss_keys = list(loss_points.keys())
    acc_keys.sort()
    loss_keys.sort()

    # 训练模型
    with tf.Session() as sess:
        sess.run(init)

        if pretrained_model_path:
            print('Loading whole model ...')

            # 加入cls之后
            variables = tf.global_variables()
            # Initialize all variables first
            sess.run(tf.variables_initializer(variables, name='init'))

            # print(variables_to_restore)
            if os.path.isdir(pretrained_model_path) and True:
                model_path = get_model_filenames_slim(pretrained_model_path)
                # else:
                #     model_path = pretrained_model_path
                print('Restoring pretrained model: %s' % model_path)

                var_keep_dic = get_variables_in_checkpoint_file(model_path)
                variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
                # variables_to_restore = [var for var in variables_to_restore if 'Resface/Conv1/Conv1' not in var.op.name]
                # variables_to_restore = [var for var in variables_to_restore if 'Resface/Conv2/Conv2' not in var.op.name]
                # variables_to_restore = [var for var in variables_to_restore if 'Resface/Conv3/Conv3' not in var.op.name]
                # variables_to_restore = [var for var in variables_to_restore if 'Group4' not in var.op.name]
                # variables_to_restore = [var for var in variables_to_restore if 'bottleneck' not in var.op.name]
                # variables_to_restore = [var for var in variables_to_restore if 'gender_bottleneck' not in var.op.name]
                # variables_to_restore = [var for var in variables_to_restore if '_sigma' not in var.op.name]
                loader_partial = tf.train.Saver(variables_to_restore, max_to_keep=1)
                loader_partial.restore(sess, model_path)

        def eval_dataset(verbose=0):
            # eval train
            # eval test
            y_test, y_cls_test, y_occ_test = [], [], []
            blur_prob_test, cls_prob_test, occ_prob_test = [], [], []
            test_values = 0
            # loss_list = [blur_predict_prob, blur_predicted]
            loss_list = [blur_predict_prob, cls_predict_prob, occ_predict_prob]
            for k in loss_keys:
                loss_list += [loss_points[k]]
            for k in acc_keys:
                loss_list += [acc_points[k]]
            all_keys = list(loss_keys) + list(acc_keys)
            acc_start_idx = len(loss_keys)

            # 属性测试集
            RESAMPLE_FLAG = True
            if RESAMPLE_FLAG:

                blur_idxes = np.concatenate((test_labels_dict['blur_pos_idxes'],
                                             test_labels_dict['blur_neg_idxes']),
                                            axis=0)
                blur_idxes = np.random.permutation(blur_idxes)
                test_file_list_blur = test_file_list[blur_idxes]

            else:
                test_file_list_blur = test_file_list

            total_batch = int(len(test_file_list_blur) // property_batch_size)
            test_file_list_blur = np.array(test_file_list_blur)[:int(property_batch_size * total_batch)]
            cls_image_list_test_ = cls_image_list_test[
                                       np.random.permutation(np.arange(0, len(cls_image_list_test)))][
                                   :int(total_batch * cls_batch_size)]

            if verbose == 0:
                total_batch = min(total_batch, 10)
            total_batch = min(total_batch, 10)

            for i in range(total_batch):
                sys.stdout.flush()
                sys.stdout.write('\rtest %d / %d ' % (i, total_batch))

                X_batch, y_batch_dict = get_data_image_and_label(
                    test_file_list_blur[i * property_batch_size: (i + 1) * property_batch_size], image_size)

                X_test_face, y_test_face = get_cls_image_test(
                    cls_image_list_test_[i * cls_batch_size: (i + 1) * cls_batch_size],
                    image_size,
                    is_training=True)

                X_batch = np.concatenate((X_batch, X_test_face), axis=0)
                # y_test_face = np.concatenate((np.ones(property_batch_size), y_test_face), axis=0)
                occlabel = (y_batch_dict['blur'] > 0.95).astype(int) + (np.abs(y_batch_dict['angle'][:,1]) > 50).astype(int)
                occlabel = 1 - (occlabel > 0).astype(int)
                y_test_face = np.concatenate((occlabel, y_test_face), axis=0)
                y_batch_dict['cls'] = y_test_face

                feed_dict = {x_input: X_batch, phase_train_placeholder: False}
                for k in label_points.keys():
                    feed_dict[label_points[k]] = y_batch_dict[k]
                values = sess.run(loss_list, feed_dict=feed_dict)

                test_values += np.array(values[3:]) / total_batch
                blur_prob_test += list(values[0][:, 1])
                cls_prob_test += list(values[1][:, 1])
                occ_prob_test += list(values[2][:, 1])
                y_test += list(y_batch_dict['blur'])
                y_cls_test += list(y_test_face)
                y_occ_test += list(y_batch_dict['occmouth'])

                # print('blur_acc', values[2])
                # disp_data(X_batch, y_batch_dict, values[1])

            s = ''
            for i in range(len(all_keys)):
                if i == acc_start_idx:
                    s += '| '
                s += '%s %.2f ' % (all_keys[i][:4], test_values[i])
            with open(os.path.join(train_dir, 'training-log.txt'), 'a') as f:
                print(s)
                f.write(s + '\n')
            if verbose > 0:
                eval_tpr_far(y_test, blur_prob_test, os.path.join(train_dir, 'training-log.txt'))
                eval_tpr_far_cls(y_cls_test, cls_prob_test, os.path.join(train_dir, 'training-log.txt'))
                eval_tpr_far_cls(1-np.array(y_occ_test), 1-np.array(occ_prob_test), os.path.join(train_dir, 'training-log.txt'),
                                 'no-occ', 'occlusion')

        for epoch in range(training_epochs):
            total_num_batch_each_epoch = int(n_sample / batch_size)
            # num_batch_each_epoch = 100

            train_values = 0
            loss_list_init = [optimizer, learning_rate]
            loss_list = loss_list_init + [total_loss]
            for k in loss_keys:
                loss_list += [loss_points[k]]
            acc_start_idx = len(loss_list) - 2
            for k in acc_keys:
                loss_list += [acc_points[k]]
            all_keys = ['total'] + list(loss_keys) + list(acc_keys)

            if epoch % 10 == 0:
                s = '-' * 20 + str(epoch) + '-' * 20
                with open(os.path.join(train_dir, 'training-log.txt'), 'a') as f:
                    print(s)
                    f.write(s + '\n')
                eval_dataset(verbose=1)
            # elif epoch % 5 == 0:
            #     eval_dataset()
            if only_eval:
                exit(0)

            if epoch % 10 == 0 and epoch > 0:
                save_variables_and_metagraph(sess, saver, train_dir, epoch)

            lr = 0
            t1 = time()
            for i in range(num_batch_each_epoch):
                RESAMPLE_FLAG = True
                if RESAMPLE_FLAG:
                    # file_list_pos = train_file_list[np.random.permutation(train_labels_dict['blur_pos_idxes'])[:(batch_size-batch_size//3)]]
                    # print(train_labels_dict.keys())
                    file_list_sample1 = train_file_list[np.random.permutation(train_labels_dict['expression_idxes_1'])[:property_batch_size // 6]]
                    # file_list_sample2 = train_file_list[np.random.permutation(train_labels_dict['gender_pos_idxes'])[:property_batch_size // 6]]
                    # file_list_sample3 = train_file_list[np.random.permutation(train_labels_dict['gender_neg_idxes'])[:property_batch_size // 6]]
                    file_list_sample2 = train_file_list[np.random.permutation(train_labels_dict['expression_idxes_1'])[:property_batch_size // 6]]
                    file_list_sample3 = train_file_list[np.random.permutation(train_labels_dict['expression_idxes_2'])[:property_batch_size // 6]]
                    file_list_sample4 = train_file_list[np.random.permutation(train_labels_dict['expression_idxes_1'])[:property_batch_size // 6]]
                    file_list_sample5 = train_file_list[np.random.permutation(train_labels_dict['expression_idxes_2'])[:property_batch_size // 6]]
                    file_list_sample6 = train_file_list[np.random.permutation(train_labels_dict['age_pos_idxes'])[:(property_batch_size - property_batch_size // 6 * 5)]]
                    file_list = np.concatenate((file_list_sample1, file_list_sample2, file_list_sample3, file_list_sample4, file_list_sample5, file_list_sample6), axis=0)
                else:
                    file_list = train_file_list[
                        np.random.permutation(np.arange(0, len(train_file_list)))[:property_batch_size]]

                # X_train_batch, y_train_dict = get_data_image_and_label(file_list, image_size, is_training=True)
                X_train_batch_0, y_train_dict_0 = get_data_image_and_label(file_list, image_size, is_training=True)

                # 加入人脸分类
                face_image_list_neg = cls_image_list_neg[
                    np.random.permutation(np.arange(0, len(cls_image_list_neg)))[:int(cls_batch_size * 0.7)]]
                face_image_list_part = cls_image_list_part[
                    np.random.permutation(np.arange(0, len(cls_image_list_part)))[:int(cls_batch_size * 0.05)]]
                face_image_list_pos = cls_image_list_pos[
                    np.random.permutation(np.arange(0, len(cls_image_list_pos)))[:int(cls_batch_size * 0.25)]]

                X_train_batch_face_neg, y_train_label_face_neg = get_cls_image_and_label(face_image_list_neg,
                                                                                         image_size, is_training=True)
                X_train_batch_face_part, y_train_label_face_part = get_cls_image_and_label(face_image_list_part,
                                                                                           image_size, is_training=True)
                X_train_batch_face_pos, y_train_label_face_pos = get_cls_image_and_label(face_image_list_pos,
                                                                                         image_size, is_training=True)

                X_train_batch_face = np.concatenate(
                    (X_train_batch_face_neg, X_train_batch_face_part, X_train_batch_face_pos), axis=0)
                y_train_label_face = np.concatenate(
                    (y_train_label_face_neg, y_train_label_face_part, y_train_label_face_pos), axis=0)

                # integrate train batch
                if len(X_train_batch_0.shape) < 4 or len(X_train_batch_face.shape) < 4 or X_train_batch_0.shape[1] != X_train_batch_face.shape[1]:
                    print('line1282, dimensions do not match, X_train_batch_0 and X_train_batch_face',
                          X_train_batch_0.shape, X_train_batch_face.shape)
                    continue
                X_train_batch = np.concatenate((X_train_batch_0, X_train_batch_face), axis=0)
                # y_train_label_face = np.concatenate((np.ones(property_batch_size), y_train_label_face), axis=0)
                occlabel = (y_train_dict_0['blur'] > 0.95).astype(int) + (np.abs(y_train_dict_0['angle'][:,1]) > 50).astype(int)
                occlabel = 1 - (occlabel > 0).astype(int)
                y_train_label_face = np.concatenate((occlabel, y_train_label_face), axis=0)
                y_train_dict_0['cls'] = y_train_label_face

                feed_dict = {x_input: X_train_batch, phase_train_placeholder: True}
                for k in label_points.keys():
                    feed_dict[label_points[k]] = y_train_dict_0[k]
                values = sess.run(loss_list, feed_dict=feed_dict)
                lr = values[1]
                train_values += np.array(values[len(loss_list_init):]) / num_batch_each_epoch

                s = '%.0f E%02.2f[%03d] lr %.1ef ' % (time()-t1, (epoch + 1) * 20 / total_num_batch_each_epoch, i, lr)
                for j in range(len(all_keys)):
                    if j == acc_start_idx:
                        s += '| '
                    s += '%s %.2f ' % (all_keys[j][:4], values[len(loss_list_init) + j])
                sys.stdout.flush()
                sys.stdout.write('\r' + s)
            sys.stdout.write('\r')

            # weight_value, bias_value = sess.run([weight['h1'], bias['h1']])
            # print(weight_value.shape, bias_value.shape)

            if epoch % 1 == 0:
                s = '%.0f E%02.2f[%03d] lr %.1ef ' % (time()-t1, (epoch + 1) * 20 / total_num_batch_each_epoch, i, lr)
                for j in range(len(all_keys)):
                    if j == acc_start_idx:
                        s += '| '
                    s += '%s %.2f ' % (all_keys[j][:4], train_values[j])
                with open(os.path.join(train_dir, 'training-log.txt'), 'a') as f:
                    print(s)
                    f.write(s + '\n')

        print('Opitimization Finished!')

        # Test
        # eval_dataset()
        # pred_values = inference_pred(X_test, logits, x_input, sess)
        # prob = pred_values[:,1]



def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []
    for v in variables:
        # exclude the first conv layer to swap RGB to BGR
        # if v.name == (self._scope + '/conv1/weights:0'):
        #     self._variables_to_fix[v.name] = v
        #     continue
        if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
    return variables_to_restore


def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def load_training_cache_with_gen(image_dir):
    print('--gen_training_cache--')
    cache_dir = os.path.join(os.path.abspath(__file__), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print('  ' + image_dir)
    name = os.path.split(image_dir)[-1] + '.npy'
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        data = np.load(path)
        return data['file_list'], data['labels_dict']
    file_list = scan_image_tree(image_dir)
    np.random.shuffle(file_list)
    file_list, labels_dict = get_data_label(file_list)
    np.save(path, {'file_list': file_list, 'labels_dict': labels_dict})
    return file_list, labels_dict


def load_training_cache(train_image_dir_list, test_image_dir_list):
    print('--gen_training_cache--')
    cache_dir = os.path.join(os.path.abspath(__file__), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    for image_dir in train_image_dir_list + test_image_dir_list:
        print('  ' + image_dir)
        name = os.path.split(image_dir)[-1] + '.npy'
        if os.path.exists(os.path.join(cache_dir, name)):
            continue
    pass


def putText(img, info=""):
    from PIL import Image, ImageDraw, ImageFont
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)
    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
    draw.text((0, 0), info, (0, 0, 255), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
    cv2_text_im = cv2.cvrColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return cv2_text_im


def load_cls_file_list(only_eval=False):
    max_count = 400000
    cls_image_list_test = []
    for name in ['negative_48_hard', 'negative_remove_pitch', 'positive_remove_yaw', 'positive_wanglizhaopian']:
        if not os.path.exists(os.path.join(cls_folder, name)):
            continue
        raw_list = np.array(os.listdir(os.path.join(cls_folder, name)))
        raw_list = raw_list[np.random.permutation(np.arange(0, len(raw_list)))][:max_count]
        cls_image_list_test += [os.path.join(cls_folder, name, e) for e in raw_list]
    cls_image_list_test = np.array(cls_image_list_test)
    if only_eval:
        return cls_image_list_test, cls_image_list_test, cls_image_list_test, cls_image_list_test

    cls_image_list_neg = []
    max_count = 400000
    for name in ['negative', 'negative_hard', 'negative_hard_angle', 'negative_aug', 'negative_IR_scence', 'negative_48_hard']:
        if not os.path.exists(os.path.join(cls_folder, name)):
            continue
        raw_list = np.array(os.listdir(os.path.join(cls_folder, name)))
        raw_list = raw_list[np.random.permutation(np.arange(0, len(raw_list)))][:max_count]
        cls_image_list_neg += [os.path.join(cls_folder, name, e) for e in raw_list]

    # max_count = 400000
    cls_image_list_part = []
    for name in ['part', 'part_remove_yaw']:
        if not os.path.exists(os.path.join(cls_folder, name)):
            continue
        raw_list = np.array(os.listdir(os.path.join(cls_folder, name)))
        raw_list = raw_list[np.random.permutation(np.arange(0, len(raw_list)))][:max_count]
        cls_image_list_part += [os.path.join(cls_folder, name, e) for e in raw_list]

    cls_image_list_pos = []
    # max_count = 400000
    for name in ['positive', 'positive_1', 'positive_2', 'positive_pub', 'positive_pub_close', 'positive_pub_cloud',
                 'positive_NUAA_clients_face', 'positive_NUAA_imposter_face', 'positive_NUAA_nomalize_client_face',
                 'positive_nir_aligned']:
        if not os.path.exists(os.path.join(cls_folder, name)):
            continue
        if name in ['positive_nir_aligned']:  # 递归
            raw_list = []
            get_images(os.path.join(cls_folder, name), raw_list)
            # raw_list = np.array(os.listdir(os.path.join(cls_folder, name)))
            raw_list = np.array(raw_list)
            raw_list = raw_list[np.random.permutation(np.arange(0, len(raw_list)))][:max_count]
        else:
            raw_list = np.array(os.listdir(os.path.join(cls_folder, name)))
            raw_list = raw_list[np.random.permutation(np.arange(0, len(raw_list)))][:max_count]
        cls_image_list_pos += [os.path.join(cls_folder, name, e) for e in raw_list]

    if len(cls_image_list_neg) is None or len(cls_image_list_part) is None or len(cls_image_list_pos) is None:
        pass

    print('cls_image_list_neg:{}'.format(len(cls_image_list_neg)))
    print('cls_image_list_part:{}'.format(len(cls_image_list_part)))
    print('cls_image_list_pos:{}'.format(len(cls_image_list_pos)))
    cls_image_list_neg = np.array(cls_image_list_neg)[:500000]
    cls_image_list_part = np.array(cls_image_list_part)[:100000]
    cls_image_list_pos = np.array(cls_image_list_pos)[:500000]
    return cls_image_list_neg, cls_image_list_part, cls_image_list_pos, cls_image_list_test


def load_face_attr_file_list(only_eval=False):
    train_file_list = []
    test_file_list = []
    train_image_dir_list = [r'E:\data\face-attr\louyu-capture\downloadimg-20180905',
                            r'E:\data\face-attr\louyu-capture\downloadimg-rec-20181025',
                            r'E:\data\face-attr\louyu-capture\downloadimg-rec-20181127',
                            r'E:\data\face-attr\louyu-capture\downloadimg-rec-20181228',
                            ]
    train_image_dir_list = [r'D:\data\face-attr\celeba-2w',
                            r'D:\data\face-attr\louyu-capture\downloadimg-20180905',
                            r'D:\data\face-attr\louyu-capture\downloadimg-rec-20181025',
                            r'D:\data\face-attr\louyu-capture\downloadimg-rec-20181127',
                            r'D:\data\face-attr\louyu-capture\downloadimg-rec-20181228',
                            ]

    train_image_dir_list = [
        r'F:\data\face-attr\affectnet\affectnet_resized',
        r'F:\data\face-attr\louyu-capture\louyu-7ch-night-20180123',
        r'F:\data\face-attr\celeba-2w',
        r'F:\data\face-attr\cacd2000-young-old',
        r'F:\data\face-attr\louyu-capture\downloadimg-20180905',
        r'F:\data\face-attr\louyu-capture\downloadimg-rec-20181025',
        r'F:\data\face-attr\louyu-capture\downloadimg-rec-20181127',
        r'F:\data\face-attr\louyu-capture\downloadimg-rec-20181228',
        r'F:\data\face-attr\zhangbiao\96\positive-with-attr-label',
        # r'F:\data\face-attr\ir_dataset',
        r'F:\data\face-attr\MAFA\train-images\images_face_crop',
    ]

    # train_image_dir_list += [r'E:\data\face-attr\celeba-20w']
    # train_image_dir_list += [r'E:\data\face-alignment\2DFaceAlignmentImageData\2DFaceAlignmentData\menpo\trainset\semifrontal']
    # train_image_dir_list += [r'D:\data\face-attr\cacd2000-young-old']
    # train_image_dir_list += [r'E:\data\face-alignment\2DFaceAlignmentImageData\2DFaceAlignmentData\multipie\semifrontal']
    # train_image_dir_list = [r'D:\data\face-attr\louyu-capture\downloadimg-20180905']
    test_image_dir_list = [
        r'F:\data\face-attr\affectnet\affectnet_resized\save\1344',
        r'F:\data\face-attr\affectnet\affectnet_resized\save\1345',
        r'F:\data\face-attr\affectnet\affectnet_resized\save\1347',
        r'F:\data\face-attr\louyu-capture\downloadimg-20180911',
        # r'D:\data\face-attr\louyu-capture\louyu-7ch-night-20180123'
    ]
    test_image_dir_list1 = [
        # r'D:\data\face-attr\louyu-capture\downloadimg-20180911',
        r'F:\data\face-attr\test\test_occlusion_002',
        # r'F:\data\face-recognition\realsense\data-labeled\ir',
        # r'F:\data\face-attr\ir_dataset\PhotoData_face_all_label_de',
        # r'E:\data\face-attr\louyu-capture\louyu-7ch-night-20180123'
    ]
    if only_eval:
        test_image_dir_list = [
            r'F:\data\face-recognition\realsense\data-labeled\ir',
            r'F:\data\face-attr\louyu-capture\downloadimg-20180911',
        ]
        train_image_dir_list = test_image_dir_list[:1]
    for image_dir in train_image_dir_list:
        print(image_dir)
        file_list = scan_image_tree(image_dir)
        np.random.shuffle(file_list)
        # file_list = file_list[:80000]
        # file_list = file_list[:10]
        train_file_list += file_list
    for image_dir in test_image_dir_list:
        print(image_dir)
        file_list = scan_image_tree(image_dir)  # 20203
        # file_list = file_list[:8000]
        file_list = file_list[:int(40 * property_batch_size + 200)]
        test_file_list += file_list
    if False:
        get_data_image_and_label(test_file_list, image_size, is_training=True)
        exit(0)
    return train_file_list, test_file_list


def main():
    reload_flag = False
    only_eval = False
    save_name_pkl = 'training_file_list_dict_ir.pkl'
    save_name_pkl = 'training_file_list_dict.pkl'
    save_name_pkl = 'training_file_list_dict_gender_expr.pkl'
    if only_eval:
        save_name_pkl = 'training_file_list_dict_eval.pkl'
        reload_flag = True

    if reload_flag:
        attr_train_file_list, attr_test_file_list = load_face_attr_file_list(only_eval)
        attr_train_file_list, attr_train_labels_dict, attr_test_file_list, attr_test_labels_dict = transform_file_list(
            attr_train_file_list, attr_test_file_list, verbose=0)
        cls_image_list_neg, cls_image_list_part, cls_image_list_pos, cls_image_list_test = load_cls_file_list(only_eval)
        training_file_list_dict = {}
        training_file_list_dict['attr_train_file_list'] = attr_train_file_list
        training_file_list_dict['attr_train_labels_dict'] = attr_train_labels_dict
        training_file_list_dict['attr_test_file_list'] = attr_test_file_list
        training_file_list_dict['attr_test_labels_dict'] = attr_test_labels_dict
        training_file_list_dict['cls_image_list_neg'] = cls_image_list_neg
        training_file_list_dict['cls_image_list_part'] = cls_image_list_part
        training_file_list_dict['cls_image_list_pos'] = cls_image_list_pos
        training_file_list_dict['cls_image_list_test'] = cls_image_list_test
        with open(save_name_pkl, 'wb') as f:
            pickle.dump(training_file_list_dict, f)
    else:
        with open(save_name_pkl, 'rb') as f:
            data = pickle.load(f)
        training_file_list_dict = data
        attr_train_file_list = training_file_list_dict['attr_train_file_list']
        attr_train_labels_dict = training_file_list_dict['attr_train_labels_dict']
        attr_test_file_list = training_file_list_dict['attr_test_file_list']
        attr_test_labels_dict = training_file_list_dict['attr_test_labels_dict']
        cls_image_list_neg = training_file_list_dict['cls_image_list_neg']
        cls_image_list_part = training_file_list_dict['cls_image_list_part']
        cls_image_list_pos = training_file_list_dict['cls_image_list_pos']
        cls_image_list_test = training_file_list_dict['cls_image_list_test']

    print('attr_train_file_list', len(attr_train_file_list))
    print('attr_test_file_list', len(attr_test_file_list))
    print('cls_image_list_neg', len(cls_image_list_neg))
    print('cls_image_list_part', len(cls_image_list_part))
    print('cls_image_list_pos', len(cls_image_list_pos))
    main_train(training_file_list_dict, only_eval=only_eval)


if __name__ == '__main__':
    # demo_camera()
    # cls_folder = r'E:\datasets\public_datasets\WIDER\96'
    # cls_folder = r'E:\data\face-attr\cls_face'
    cls_folder = r'F:\data\face-attr\zhangbiao\96'
    # train_file_list_ = np.loadtxt('train_file_list.txt', dtype=str)
    # test_file_list = np.loadtxt('test_file_list.txt', dtype=str)
    # with open('train_labels_dict.pickle', 'rb') as handle:
    #     train_labels_dict = pickle.load(handle)
    # with open('test_labels_dict.pickle', 'rb') as handle:
    #     test_labels_dict = pickle.load(handle)
    # pass

    main()
