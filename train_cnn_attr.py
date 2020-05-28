# coding=utf-8
import tensorflow
import sys
# from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.svm import LinearSVC
# from theano.gof.cmodule import importlib
import importlib
import pickle
import time
# import networks
# from face_attr.corrupt_image import motion_blur, defocus_blur
from face_data_augment import image_augment_cv, random_bbox
import common

# 设置模型参数
from common import scan_image_tree
from models import hourglass_arg_scope_tf
from tensorflow.python import pywrap_tensorflow

# learning_rate_init = 0.00003
learning_rate_init = 0.000013
learning_rate_decay = 0.93
training_epochs = 8000
num_batch_each_epoch = 40

batch_size = 512
# batch_size = 600
cls_batch_size = 0
antispoof_batch_size = 0
# batch_size = 60
# cls_batch_size = 30
attr_batch_size = batch_size - cls_batch_size - antispoof_batch_size

weight_decay = 1e-6
# opt_name = 'mom'
opt_name = 'adam'

# model_def = 'nets.mobilenet_v2'
# model_def = 'nets.irv1_stem_flatten'
# model_def = 'nets.irv1_small3'
# model_def = 'nets.irv1_stem_concat'
model_def = 'nets.resface12_relu_avg'
# model_def = 'nets.irv1_stem'

image_size = 96
# image_size = 48
# image_size = 64


train_dir = r"running_tmp\20200518-integrate-96x96-norm256-attr-antispoof"

pretrained_model_path = r"running_tmp\20200315-integrate-96x96-norm256-attr-antispoof"
# pretrained_model_path = train_dir

logs_train_dir = train_dir + r'\model_save_' + model_def

use_sigma_loss_weight = False
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
    'pts5': loss_scale * 0.5,
    'glasses': loss_scale * 0.1,
    'gender': 0.1,
    'occlusion': loss_scale * 0.1,
    'occleye': loss_scale * 0.1,
    'occreye': loss_scale * 0.1,
    'occmouth': loss_scale * 0.1,
    # 'cls': loss_scale * 0.1,
    # 'antispoof': loss_scale * 1.5
}


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
        cnt_p = 0.
        cnt_n = 0.
        idx = np.argsort(prob)
        total_p = np.sum(y == 0)
        total_n = np.sum(y == 1)
        s = 'total_p %d total_n %d' % (total_p, total_n)
        print(s)
        f.write(s + '\n')
        temp = -1
        for cnt, i in enumerate(idx):
            if y[i] == 1:
                cnt_n += 1
            else:
                cnt_p += 1
            tpr = cnt_n / total_n
            far = cnt_p / total_p
            disp_far_list = [1 / total_p, 0.0001, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            if cnt_p > temp:
                for disp_far in disp_far_list:
                    if cnt_p == int(disp_far * total_p):
                        temp = cnt_p
                        s = 'cnt_p: %3d  cnt_n: %5d  label: %d  blur: %f  clean: %f  tpr: %f  far: %f' % (
                            cnt_p, cnt_n, int(y[i]), prob[i], 1 - prob[i], tpr, disp_far)
                        print(s)
                        f.write(s + '\n')
                        break


def eval_tpr_far_cls(y, prob, filename_output):
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
        total_n = np.sum(y == 0)
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

            disp_far_list = [1 / total_n, 0.0001, 0.0003, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
            if cnt_n > temp:
                for disp_far in disp_far_list:
                    if cnt_n == int(disp_far * total_n):
                        temp = cnt_n
                        # tpr: 通过率
                        s = 'cnt_p: %3d  cnt_n: %5d  label: %d  face: %f  noface: %f  tpr(cnt_p/total_p): %f  far(cnt_n/total_n): %f' % (
                            cnt_p, cnt_n, int(y[i]), prob[i], 1 - prob[i], tpr, disp_far)
                        print(s)
                        f.write(s + '\n')
                        break


def eval_tpr_far_antispoof(y, prob, filename_output):
    '''
    :param y:  0 is live, 1(1 is phone, 2 is paper) 测试时候关注live
    :param prob: live score
    :param filename_output:
    :return:
    '''
    y = 1 - np.array(y)  # live0->1
    prob = np.array(prob)
    print(filename_output)
    with open(filename_output, 'a') as f:
        cnt_p = 0.
        cnt_n = 0.
        idx = np.argsort(prob)
        total_p = np.sum(y == 1)  # live0 ==>1 正
        total_n = np.size(y) - total_p
        s = 'antispoof-live: total_p %d total_n %d' % (total_p, total_n)
        f.write(s + '\n')
        temp = -1
        for cnt, i in enumerate(idx):
            # live0 ==>1 正
            if y[i] == 1:
                cnt_p += 1
            else:
                cnt_n += 1

            if total_n == 0:
                tpr = 1
            else:
                tpr = cnt_n / total_n

            # far = cnt_p / total_p
            disp_far_list = [1 / total_p, 0.0001, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            if cnt_p > temp:
                for disp_far in disp_far_list:
                    if cnt_p == int(disp_far * total_p):
                        temp = cnt_p
                        s = 'cnt_p: %3d  cnt_n: %5d  label: %d live: %f  anti: %f  tpr: %f  far: %f' % (
                            cnt_p, cnt_n, int(y[i]), prob[i], 1 - prob[i], tpr, disp_far)
                        print(s)
                        f.write(s + '\n')
                        break


def eval_tpr_far_antispoof_phone(y, prob, filename_output):
    '''
    :param y:  0 is live, 1(1 is phone, 2 is paper) 测试时候关注live
    :param prob:
    :param filename_output:
    :return:
    '''
    y = np.array(y)
    prob = np.array(prob)
    print(filename_output)
    with open(filename_output, 'a') as f:
        cnt_p = 0.
        cnt_n = 0.
        idx = np.argsort(prob)
        total_p = np.sum(y == 1)  # phone 1 正
        total_n = np.size(y) - total_p
        s = 'antispoof-phone: total_p %d total_n %d' % (total_p, total_n)
        f.write(s + '\n')
        temp = -1
        for cnt, i in enumerate(idx):
            # phone 1 正
            if y[i] == 1:
                cnt_p += 1
            else:
                cnt_n += 1
            if total_n == 0:
                tpr = 1
            else:
                tpr = cnt_n / total_n

            # far = cnt_p / total_p
            disp_far_list = [1 / total_p, 0.0001, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            if cnt_p > temp:
                for disp_far in disp_far_list:
                    if cnt_p == int(disp_far * total_p):
                        temp = cnt_p
                        s = 'cnt_p: %3d  cnt_n: %5d  label: %d  phone: %f not_phone: %f  tpr: %f  far: %f' % (
                            cnt_p, cnt_n, int(y[i]), prob[i], 1 - prob[i], tpr, disp_far)
                        print(s)
                        f.write(s + '\n')
                        break


def eval_tpr_far_antispoof_paper(y, prob, filename_output):
    '''
    :param y:  0 is live, 1(1 is phone, 2 is paper) 测试时候关注live
    :param prob:
    :param filename_output:
    :return:
    '''
    y = 3 - np.array(y)  # paper2  ==》 1 正
    prob = np.array(prob)
    print(filename_output)
    with open(filename_output, 'a') as f:
        cnt_p = 0.
        cnt_n = 0.
        idx = np.argsort(prob)
        total_p = np.sum(y == 1)  # paper2  ==》 1 正
        total_n = np.size(y) - total_p
        s = 'antispoof-paper: total_p %d total_n %d' % (total_p, total_n)
        f.write(s + '\n')
        temp = -1
        for cnt, i in enumerate(idx):
            # paper2  ==》 1 正
            if y[i] == 1:
                cnt_p += 1
            else:
                cnt_n += 1
            if total_n == 0:
                tpr = 1
            else:
                tpr = cnt_n / total_n

            # far = cnt_p / total_p
            disp_far_list = [1 / total_p, 0.0001, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            if cnt_p > temp:
                for disp_far in disp_far_list:
                    if cnt_p == int(disp_far * total_p):
                        temp = cnt_p
                        s = 'cnt_p: %3d  cnt_n: %5d  label: %d  paper: %f  not_paper: %f  tpr: %f  far: %f' % (
                            cnt_p, cnt_n, int(y[i]), prob[i], 1 - prob[i], tpr, disp_far)
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


def _build_network(model_def, inputs, is_training=True, n_landmarks=68):
    image_batch = tf.identity(inputs, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')

    print('Building training graph')

    weight_decay = 1e-6
    # model_def = 'nets.irv1_stem'
    print(model_def)
    network = importlib.import_module(model_def)
    print('weight_decay', weight_decay)
    print(network)

    # Build the inference graph
    drop_out = 0.6
    # drop_out = 1.0
    prelogits, states = network.inference(image_batch, drop_out, phase_train=is_training, weight_decay=weight_decay)
    if 'bottleneck' in model_def or 'resface' in model_def:
        bottleneck = prelogits
    else:
        bottleneck = slim.fully_connected(prelogits, 128, activation_fn=None,
                                          scope='Bottleneck', reuse=False)
    # states['bottleneck'] = bottleneck
    # print(bottleneck.get_shape())

    # embeddings = tf.nn.l2_normalize(bottleneck, 1, 1e-10, name='embeddings')
    # states['embeddings'] = embeddings

    prediction = slim.fully_connected(bottleneck, n_landmarks * 2, activation_fn=None, scope='prediction', reuse=False)
    states['pts68_predictions'] = prediction
    print(prediction, prediction.get_shape(), prediction.dtype)

    # net = tf.concat([bottleneck, prediction], axis=1)
    # angle_hidden = slim.fully_connected(net, 128, activation_fn=None, scope='angle_hidden', reuse=False)
    # pred_logits_angle = slim.fully_connected(angle_hidden, 3, activation_fn=None, scope='pred_angle', reuse=False)
    # states['angle_predictions'] = pred_logits_angle
    #
    # print(pred_logits_angle, pred_logits_angle.get_shape(), pred_logits_angle.dtype)
    # states['pred_logits_angle'] = pred_logits_angle

    return None, states


def build_network(images_input, model_def=model_def, n_landmarks=68, is_training=False):
    with tf.variable_scope('net'):
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
            with slim.arg_scope(hourglass_arg_scope_tf()):
                _, end_points = _build_network(model_def, images_input, is_training=is_training,
                                               n_landmarks=n_landmarks)
                with tf.variable_scope('net', 'MultiScaleLayer'):
                    def get_tensors_by_avgpool(net_temp):
                        net_temp = slim.avg_pool2d(net_temp, net_temp.get_shape()[1:3], padding='VALID',
                                                   scope='AvgPool')
                        # net_temp = slim.avg_pool2d(net_temp, net_temp.get_shape()[1:3], padding='SAME', scope='AvgPool')
                        net_temp = slim.flatten(net_temp)
                        return net_temp

                with tf.variable_scope('FaceAttr'):
                    net_avgpool = get_tensors_by_avgpool(end_points['Conv4'])
                    net = slim.fully_connected(net_avgpool, 128, scope='age_bottleneck', reuse=False)

                    end_points['age_predictions'] = slim.fully_connected(net, 1, activation_fn=None,
                                                                         scope='age_predictions', reuse=False)
                    end_points['gender_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                            scope='gender_predictions', reuse=False)

                    if 'cls' in loss_weight.keys():
                        # end_points['cls_predictions'] = slim.fully_connected(net, num_outputs=2,
                        #                                                     activation_fn=None,
                        #                                                     scope='cls_predictions', reuse=False)

                        end_points['cls_predictions'] = slim.fully_connected(net, num_outputs=2,
                                                                             activation_fn=tf.nn.softmax,
                                                                             scope='cls_predictions', reuse=False)

                    # add
                    end_points['angle_predictions'] = slim.fully_connected(net, 3,
                                                                           activation_fn=None,
                                                                           scope='angle_predictions',
                                                                           reuse=False)

                    end_points['pts5_predictions'] = slim.fully_connected(net, 5 * 2, activation_fn=None,
                                                                          scope='pts5_predictions', reuse=False)

                    end_points['pts4_predictions'] = slim.fully_connected(net, 4 * 2, activation_fn=None,
                                                                          scope='pts4_predictions', reuse=False)

                    end_points['blur_predictions'] = slim.fully_connected(net, 2, activation_fn=None,
                                                                          scope='blur_predictions',
                                                                          reuse=False)
                    # end_points['blur_predictions'] = slim.fully_connected(net, 2, activation_fn=tf.nn.softmax,
                    #                                                           scope='blur_predictions',
                    #                                                           reuse=False)

                    end_points['glasses_predictions'] = slim.fully_connected(net, 3, activation_fn=None,
                                                                             scope='glasses_predictions', reuse=False)
                    if 'antispoof' in loss_weight.keys():
                        # added on 2020/02/28
                        end_points['antispoof_predictions'] = slim.fully_connected(net, num_outputs=3,
                                                                                   activation_fn=None,
                                                                                   scope='antispoof_predictions',
                                                                                   reuse=False)

                    with tf.variable_scope('FaceAttrOther'):
                        net = slim.fully_connected(net_avgpool, 128, scope='attr_bottleneck', reuse=False)

                        end_points['beauty_predictions'] = slim.fully_connected(net, 1, activation_fn=None,
                                                                                scope='beauty_predictions',
                                                                                reuse=False)
                        end_points['expression_predictions'] = slim.fully_connected(net, 3, activation_fn=None,
                                                                                    scope='expression_predictions',
                                                                                    reuse=False)
                        end_points['race_predictions'] = slim.fully_connected(net, 4, activation_fn=None,
                                                                              scope='race_predictions', reuse=False)

                    return end_points


def get_acc_points(label_points, end_points, attr_valid_inds, cls_valid_inds, antispoof_valid_inds):
    acc_list = ['blur', 'expression', 'race', 'glasses', 'gender', 'cls', 'antispoof']
    acc_points = {}
    pred_points = {}
    for name in acc_list:
        if name in label_points.keys():
            if 'cls' in name:
                logits = tf.gather(end_points[name + '_predictions'], cls_valid_inds)
                # predicted = tf.cast(tf.arg_max(logits, 1), tf.int32)
                # accuracy_clf = tf.reduce_mean(tf.cast(tf.equal(predicted, label_points[name]), tf.float32))
                predicted = tf.cast(tf.arg_max(logits, 1), tf.float32)
                pass
            elif 'antispoof' in name:
                logits = tf.gather(end_points[name + '_predictions'], antispoof_valid_inds)
                predicted = tf.cast(tf.arg_max(logits, 1), tf.int32)
            else:
                logits = tf.gather(end_points[name + '_predictions'], attr_valid_inds)
                # predict_prob = tf.nn.softmax(logits)
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
    # loss_points['blur'] = cost_blur
    loss_points['blur'] = cost_blur * 1.5  # update teo-20190429

    # landmark cost
    m = 10
    pts68_pred = tf.gather(end_points['pts68_predictions'], attr_inds)
    # l2norm = slim.losses.mean_squared_error(predictions*m, gt_lms*m)
    cost_pts68 = slim.losses.absolute_difference(pts68_pred * m, label_points['pts68'] * m)
    loss_points['pts68'] = cost_pts68

    if 'angle' in label_points:
        scale = 5
        angle_pre = tf.gather(end_points['angle_predictions'], attr_inds)
        # loss_points['angle'] = 0.3 * slim.losses.absolute_difference(angle_pre / 10,
        #                                                              label_points['angle'] / 10)
        loss_points['angle'] = 1.5 * slim.losses.absolute_difference(angle_pre / scale,
                                                                     label_points[
                                                                         'angle'] / scale)  # update teo-20190429
    if 'age' in label_points:
        age_pre = tf.gather(end_points['age_predictions'], attr_inds)
        scale = 30
        # print(age_pre.get_shape())
        # print(label_points['age'].get_shape())
        # loss_points['age'] = 10.0 * slim.losses.absolute_difference(tf.reshape(age_pre, (-1,)) / 100,label_points['age'] / 100)
        loss_points['age'] = 3.0 * slim.losses.absolute_difference(tf.reshape(age_pre, (-1,)) / scale,
                                                                   label_points['age'] / scale)

    if 'beauty' in label_points:
        beauty_pred = tf.reshape(tf.gather(end_points['beauty_predictions'], attr_inds), (-1,))
        # loss_points['beauty'] = 10.0 * slim.losses.absolute_difference(beauty_pred / 100, label_points['beauty'] / 100)
        loss_points['beauty'] = slim.losses.absolute_difference(beauty_pred / 100, label_points['beauty'] / 100)

    if 'expression' in label_points:
        expression_pre = tf.gather(end_points['expression_predictions'], attr_inds)
        # loss_points['expression'] = 0.9 * tf.reduce_mean(slim.losses.sparse_softmax_cross_entropy(logits=expression_pre, labels=label_points['expression']))
        loss_points['expression'] = 0.3 * tf.reduce_mean(
            slim.losses.sparse_softmax_cross_entropy(logits=expression_pre, labels=label_points['expression']))

    if 'race' in label_points:
        race_pre = tf.gather(end_points['race_predictions'], attr_inds)
        # loss_points['race'] = 0.9 * tf.reduce_mean(slim.losses.sparse_softmax_cross_entropy(logits=race_pre, labels=label_points['race']))
        loss_points['race'] = 0.1 * tf.reduce_mean(
            slim.losses.sparse_softmax_cross_entropy(logits=race_pre, labels=label_points['race']))

    if 'pts4' in label_points:
        pts4_pre = tf.gather(end_points['pts4_predictions'], attr_inds)
        # loss_points['pts4'] = 0.9 * slim.losses.absolute_difference(pts4_pre * 10,label_points['pts4'] * 10)
        scale = 13
        loss_points['pts4'] = 2.5 * slim.losses.absolute_difference(pts4_pre * scale,
                                                                    label_points['pts4'] * scale)  # 更新teo-20190429

    if 'pts5' in label_points:
        scale = 13
        pts5_pre = tf.gather(end_points['pts5_predictions'], attr_inds)
        # loss_points['pts4'] = 0.9 * slim.losses.absolute_difference(pts4_pre * 10,label_points['pts4'] * 10)
        loss_points['pts5'] = 2.5 * slim.losses.absolute_difference(pts5_pre * scale,
                                                                    label_points['pts5'] * scale)  # 更新teo-20190429

    if 'glasses' in label_points:
        glasses_pre = tf.gather(end_points['glasses_predictions'], attr_inds)
        loss_points['glasses'] = 0.9 * tf.reduce_mean(
            slim.losses.sparse_softmax_cross_entropy(logits=glasses_pre, labels=label_points['glasses']))

    if 'gender' in label_points:
        gender_pre = tf.gather(end_points['gender_predictions'], attr_inds)
        # loss_points['gender'] = 1.0 * tf.reduce_mean(slim.losses.softmax_cross_entropy(logits=gender_pre, onehot_labels=tf.one_hot(label_points['gender'], 2),label_smoothing=0.1))
        # loss_points['gender'] = 1.0 * tf.reduce_mean(slim.losses.softmax_cross_entropy(logits=gender_pre, onehot_labels=tf.one_hot(label_points['gender'], 2),label_smoothing=0.1))
        loss_points['gender'] = 1.0 * tf.reduce_mean(
            slim.losses.softmax_cross_entropy(logits=gender_pre, onehot_labels=tf.one_hot(label_points['gender'], 2),
                                              label_smoothing=0.1))

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    assert len(regularization_losses) > 0
    regularization_losses = [tf.cast(l, tf.float32) for l in regularization_losses]
    regularizers = tf.add_n(regularization_losses, name='total_loss')
    loss_points['regu'] = regularizers
    return loss_points


def build_loss_face_cls(cls_prob, label):
    # cls_prob = tf.nn.softmax(cls_prob)

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
    num_keep_radio = 0.7
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def get_test_image_list(root_folder):
    folder_list = os.listdir(root_folder)
    image_list = []
    for folder in folder_list:
        path = os.path.join(root_folder, folder)
        temp = [os.path.join(path, e) for e in os.listdir(path)]
        image_list += temp
    return image_list


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
        else:
            # print('Variables not exist:{}'.format(v.name))
            pass
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


def build_input_placeholder(batch_size, cls_batch_size, antispoof_batch_size):
    label_points = {}
    # label分为两部分，属性部分，
    property_batch_size = batch_size - cls_batch_size - antispoof_batch_size
    label_points['blur'] = tf.placeholder('int32', [property_batch_size, ])
    label_points['pts68'] = tf.placeholder('float', [property_batch_size, 68 * 2])
    label_points['angle'] = tf.placeholder('float', [property_batch_size, 3])
    label_points['pts4'] = tf.placeholder('float', [property_batch_size, 4 * 2])
    label_points['pts5'] = tf.placeholder('float', [property_batch_size, 5 * 2])

    label_points['glasses'] = tf.placeholder('int32', [property_batch_size, ])
    label_points['age'] = tf.placeholder('float', [property_batch_size, ])
    label_points['beauty'] = tf.placeholder('float', [property_batch_size, ])
    label_points['expression'] = tf.placeholder('int32', [property_batch_size, ])
    label_points['race'] = tf.placeholder('int32', [property_batch_size, ])

    label_points['gender'] = tf.placeholder('int32', [property_batch_size, ])

    if 'cls' in loss_weight.keys():
        label_points['cls'] = tf.placeholder('float32', [cls_batch_size, ])
    if 'antispoof' in loss_weight.keys():
        label_points['antispoof'] = tf.placeholder('int32', [antispoof_batch_size, ])

    return label_points


def train(only_eval=False):
    import config
    from data.attr_dataset import AttrFacefDB
    # from data.antispoof_dataset import AntiSpoofDB
    # from data.frontal_cls_dataset import FrontalFacefDB

    tmp_folder = r'data/data_train_attr_temp'
    tmp_folder = os.path.join(config.APP_ROOT, 'data', 'data_train_attr_temp')
    attr_face_db = AttrFacefDB(tmp_folder, attr_batch_size)

    # tmp_folder = r'data/data_train_temp'
    # tmp_folder = os.path.join(config.APP_ROOT, 'data', 'data_train_temp')
    # antispoof_face_db = AntiSpoofDB(tmp_folder, antispoof_batch_size)
    # tmp_folder = r'data/data_train_cls_temp'
    # tmp_folder = os.path.join(config.APP_ROOT, 'data', 'data_train_cls_temp')
    # cls_face_db = FrontalFacefDB(tmp_folder, cls_batch_size)

    ####################################################################################################################
    n_sample = 100000
    phase_train_placeholder = True
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    x_input = tf.placeholder('float', [batch_size, image_size, image_size, 3])

    attr_valid_inds = tf.range(0, attr_batch_size)
    cls_valid_inds = tf.range(attr_batch_size, attr_batch_size + cls_batch_size)
    antispoof_valid_inds = tf.range(attr_batch_size + cls_batch_size, batch_size)

    label_points = build_input_placeholder(batch_size=batch_size,
                                           cls_batch_size=cls_batch_size,
                                           antispoof_batch_size=antispoof_batch_size)
    end_points = build_network(x_input, is_training=phase_train_placeholder)

    acc_points, pred_points = get_acc_points(label_points, end_points,
                                             attr_valid_inds,
                                             cls_valid_inds,
                                             antispoof_valid_inds)

    # blur pre
    label_blur = label_points['blur']
    logits_blur = tf.gather(end_points['blur_predictions'], attr_valid_inds)
    blur_predict_prob = tf.nn.softmax(logits_blur)
    # blur_predicted = tf.cast(tf.arg_max(logits_blur, 1), tf.int32)
    # blur_accuracy_clf = tf.reduce_mean(tf.cast(tf.equal(blur_predicted, label_blur), tf.float32))

    # attr loss
    loss_points = build_loss(end_points, label_points, attr_valid_inds)

    # total loss start ############################################################################################

    def calc_multi_loss(loss_list):
        multi_loss_layer = MultiLossLayer(loss_list)
        return multi_loss_layer.get_loss()

    if use_sigma_loss_weight:
        total_loss = calc_multi_loss([v for k, v in loss_points.items() if k != 'regu'])
        total_loss += loss_points['regu']
    else:
        loss_weight['regu'] = 1.
        # total loss
        total_loss = 0
        for k, v in loss_points.items():
            total_loss += loss_weight[k] * v

    # total loss end ############################################################################################

    step = tf.Variable(0, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(
        learning_rate_init,  # Base learning rate.
        step * batch_size,  # Current index into the dataset.
        n_sample,  # Decay step.
        learning_rate_decay,  # Decay rate.
        staircase=True)

    trainable_scopes = ['net/blur_bottleneck', 'net/blur_cpredictions']
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
    # end################################################################################################################

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    with open(os.path.join(train_dir, 'training-log.txt'), 'a') as f:
        print('training')

    # 初始化所有变量
    init = tf.initialize_all_variables()
    variables_to_restore = slim.get_model_variables()
    saver = tf.train.Saver(variables_to_restore, max_to_keep=5)

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
                loader_partial = tf.train.Saver(variables_to_restore, max_to_keep=1)
                loader_partial.restore(sess, model_path)

        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        for key_, item_ in acc_points.items():
            tf.summary.scalar(key_, item_)
        for key_, item_ in loss_points.items():
            tf.summary.scalar(key_, item_)

        # summary_op = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        test_total_batch = 50

        def eval_dataset(verbose=0):
            # eval test ouput
            y_test, y_cls_test, y_antispoof_test = [], [], []
            blur_prob_test = []
            test_values = 0
            # loss_list = [blur_predict_prob, blur_predicted]
            loss_list = [blur_predict_prob]
            for k in loss_keys:
                loss_list += [loss_points[k]]
            for k in acc_keys:
                loss_list += [acc_points[k]]
            all_keys = list(loss_keys) + list(acc_keys)
            acc_start_idx = len(loss_keys)

            # test_total_batch = 50
            for i in range(test_total_batch):
                sys.stdout.flush()
                sys.stdout.write('\rtest %d / %d ' % (i, test_total_batch))
                # attr batch
                X_batch_test, y_batch_dict = attr_face_db.get_next_test_batch_data()

                X_batch = X_batch_test
                if len(X_batch) != batch_size:
                    print("len(X_batch) != batch_size: {} ->{} ".format(len(X_batch), batch_size))
                    continue

                feed_dict = {x_input: X_batch, phase_train_placeholder: False}
                for k in label_points.keys():
                    feed_dict[label_points[k]] = y_batch_dict[k]
                values = sess.run(loss_list, feed_dict=feed_dict)

                test_values += np.array(values[2:]) / test_total_batch
                blur_prob_test += list(values[0][:, 1])
                y_test += list(y_batch_dict['blur'])

            s = ''
            for i in range(len(all_keys)):
                if i == acc_start_idx:
                    s += '| '
                try:
                    s += '%s %.3f ' % (all_keys[i][:4], test_values[i])
                except Exception as e:
                    print(e)
                    print(all_keys)
                    # print(test_values)
                    print('*' * 20)
            print(s)

            with open(os.path.join(train_dir, 'training-log.txt'), 'a') as f:
                print(s)
                f.write(s + '\n')
            if verbose > 0:
                eval_tpr_far(y_test, blur_prob_test, os.path.join(train_dir, 'training-log.txt'))


        # 训练
        loss_list_init = [optimizer, learning_rate]
        loss_list = loss_list_init + [total_loss]
        for k in loss_keys:
            loss_list += [loss_points[k]]
        acc_start_idx = len(loss_list) - 2
        for k in acc_keys:
            loss_list += [acc_points[k]]
        all_keys = ['total'] + list(loss_keys) + list(acc_keys)
        for epoch in range(training_epochs):
            total_num_batch_each_epoch = int(n_sample / batch_size)

            if epoch % 1 == 0:
                test_total_batch = 10
                if np.random.random() < 0.1:
                    test_total_batch = 40
                eval_dataset(verbose=1)

            if only_eval:
                exit(0)

            if epoch % 1 == 0 and epoch > 0:
                save_variables_and_metagraph(sess, saver, train_dir, epoch)

            lr = 0
            train_values = 0
            for i in range(num_batch_each_epoch):
                print("\nprocess epoch:{}/{}".format(i, num_batch_each_epoch))
                # attr
                X_train_batch_attr, y_train_label_attr = attr_face_db.get_next_train_batch_data()

                # # 加入人脸分类
                # X_train_batch_cls, y_train_label_cls = cls_face_db.get_next_train_batch_data()
                # # 加入antispoof数据
                # X_train_batch_antispoof, y_train_label_antispoof = antispoof_face_db.get_next_train_batch_data()

                # integrate train batch
                # X_train_batch = np.concatenate((X_train_batch_attr, X_train_batch_cls, X_train_batch_antispoof), axis=0)
                # y_train_label_attr['cls'] = y_train_label_cls
                # y_train_label_attr['antispoof'] = y_train_label_antispoof

                # if len(X_train_batch_attr) != attr_batch_size \
                #         or len(X_train_batch_cls) != cls_batch_size \
                #         or len(y_train_label_antispoof) != antispoof_batch_size:
                #     print('len X_train_batch_attr:{}'.format(len(X_train_batch_attr)))
                #     print('len X_train_batch_cls:{}'.format(len(X_train_batch_cls)))
                #     print('len y_train_label_antispoof:{}'.format(len(y_train_label_antispoof)))

                if len(X_train_batch_attr) != attr_batch_size:
                    continue

                feed_dict = {x_input: X_train_batch_attr, phase_train_placeholder: True}
                for k in label_points.keys():
                    feed_dict[label_points[k]] = y_train_label_attr[k]

                values = sess.run(loss_list, feed_dict=feed_dict)
                lr = values[1]
                train_values += np.array(values[len(loss_list_init):]) / num_batch_each_epoch

                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                s = '[train] '
                s += 'time:%s %d: E%02.2f[%03d] lr %.1ef ' % (loc_time,
                                                              epoch,
                                                              (epoch + 1) * 20 / total_num_batch_each_epoch,
                                                              i,
                                                              lr)
                for j in range(len(all_keys)):
                    if j == acc_start_idx:
                        s += '| '
                    s += '%s %.3f ' % (all_keys[j][:4], values[len(loss_list_init) + j])
                sys.stdout.flush()
                sys.stdout.write('\r' + s)
                if i % 5 == 0:
                    with open(os.path.join(train_dir, 'training-log.txt'), 'a') as f:
                        f.write(s + '\n')
            sys.stdout.write('\r')

            if epoch % 1 == 0:
                # s = 'E%02.2f[%03d] lr %.1ef ' % ((epoch + 1) * 20 / total_num_batch_each_epoch, i, lr)
                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                s = '[train] '
                s += 'time:%s %d: E%02.2f[%03d] lr %.1ef ' % (loc_time,
                                                              epoch,
                                                              (epoch + 1) * 20 / total_num_batch_each_epoch,
                                                              i,
                                                              lr)

                for j in range(len(all_keys)):
                    if j == acc_start_idx:
                        s += '| '
                    s += '%s %.3f ' % (all_keys[j][:4], train_values[j])
                with open(os.path.join(train_dir, 'training-log.txt'), 'a') as f:
                    print(s)
                    f.write(s + '\n')

                # save summary
                # summary_str = sess.run(summary_op, feed_dict=feed_dict)
                # train_writer.add_summary(summary_str, step)

        print('Opitimization Finished!')


class MultiLossLayer():
    def __init__(self, loss_list):
        self._loss_list = loss_list
        self._sigmas_sq = []
        for i in range(len(self._loss_list)):
            self._sigmas_sq.append(slim.variable('Sigma_sq_' + str(i), dtype=tf.float32, shape=[],
                                                 initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))

    def get_loss(self):
        loss = 0
        for i in range(0, len(self._sigmas_sq)):
            factor = tf.div(1.0, tf.multiply(2.0, tf.maximum(1e-6, self._sigmas_sq[i])))
            loss += tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i]))
        return loss


if __name__ == '__main__':
    # train(only_eval=True)
    train(only_eval=False)
