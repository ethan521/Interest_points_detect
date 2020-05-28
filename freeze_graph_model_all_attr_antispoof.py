from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import networks
from common import get_model_filenames_slim
import tensorflow.contrib.slim as slim
import importlib
import re
import numpy as np

import config
from train_cnn_attr_lock_pts5_antispoof import build_network

epoch = '25'
# epoch = '64'
dater = '20200315'
model_dir = r'..\face_attr\running_tmp'
model_path = r'..\running_tmp\2020303-integrate-96x96-cls-attr-antispoof\model.ckpt-{}'.format(epoch)
model_path = config.APP_ROOT + r'\running_tmp\20200307-integrate-96x96-cls-cls4-attr-antispoof\model.ckpt-{}'.format(
    epoch)
model_path = config.APP_ROOT + r'\running_tmp\20200308-integrate-96x96-norm256-attr-antispoof\model.ckpt-{}'.format(
    epoch)
model_path = config.APP_ROOT + r'\running_tmp\20200309-integrate-96x96-norm256-attr-antispoof\model.ckpt-{}'.format(
    epoch)
model_path = config.APP_ROOT + r'\running_tmp\20200310-integrate-96x96-norm256-attr-antispoof\model.ckpt-{}'.format(
    epoch)
model_path = config.APP_ROOT + r'\running_tmp\20200311-integrate-96x96-norm256-attr-antispoof\model.ckpt-{}'.format(
    epoch)
model_path = config.APP_ROOT + r'\running_tmp\20200312-integrate-96x96-norm256-attr-antispoof\model.ckpt-{}'.format(
    epoch)
model_path = config.APP_ROOT + r'\running_tmp\20200314-integrate-96x96-norm256-attr-antispoof\model.ckpt-{}'.format(
    epoch)
model_path = config.APP_ROOT + r'\running_tmp\{}-integrate-96x96-norm256-attr-antispoof\model.ckpt-{}'.format(
    dater, epoch)
# if not os.path.exists(model_path):
#     print('the path not exist :{}'.format(model_path))
#     exit()
output_file = os.path.split(model_path)[0] + '/' + os.path.split(model_path)[-1].replace(".ckpt-",
                                                                                         "_{}_res12_attr".format(
                                                                                             dater)) + "_gn_rgb2x96x96x3_spoof.pb"


def main():
    model_def = 'nets.resface12_relu_avg'
    print(output_file)
    image_size = 96

    with tf.Graph().as_default() as g:
        # images_input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3), name='input_images')
        # images_input = tf.placeholder(tf.float32, shape=(1, image_size, image_size, 3), name='input_images')
        images_input = tf.placeholder(tf.float32, shape=(2, image_size, image_size, 3), name='input_images')
        end_points = build_network(images_input, is_training=False)
        # pred_list = ['blur', 'cls', 'angle', 'pts4', 'pts68']
        # pred_list = ['blur', 'cls', 'angle', 'pts4', 'pts5', 'pts68']
        # pred_list = ['blur', 'cls', 'angle', 'pts4', 'pts5', 'pts68', 'age', 'gender', 'glasses']
        # pred_list = ['blur', 'cls', 'angle', 'age', 'gender', 'pts4']
        # pred_list = ['blur', 'cls', 'angle']
        # pred_list = ['blur', 'cls', 'angle',  'glasses']  # 2，2，3，3
        # pred_list = ['blur', 'cls', 'angle', 'pts4', 'pts5']
        # pred_list = ['blur', 'cls', 'angle', 'pts4']
        pred_list = ['antispoof']
        pred_list = ['cls',
                     'blur',
                     'gender',
                     'glasses',
                     'antispoof',
                     'angle',
                     'pts4',
                     # 'pts5',
                     # 'pts68',
                     'age',
                     # 'beauty',
                     'expression',
                     'race']
        # pred_list = [ 'blur', 'cls', 'angle', 'pts4', 'antispoof' ]
        pred_list = sorted(pred_list)
        collect_res = []
        for k in pred_list:
            if k in ['cls']:  # 已经过softmax
                collect_res.append(end_points[k + '_predictions'])
            if k in ['blur', 'gender', 'glasses', 'antispoof', 'expression', 'race']:
                collect_res.append(tf.nn.softmax(end_points[k + '_predictions']))

            if k in ['angle', 'pts4', 'pts5', 'pts68', 'age', 'beauty']:
                collect_res.append(end_points[k + '_predictions'])

        res = tf.concat(values=collect_res, axis=1, name='res_prediction')

        variables_to_restore = slim.get_variables_to_restore()
        print(variables_to_restore)
        saver = tf.train.Saver(variables_to_restore)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=g)

        saver.restore(sess, model_path)

        # Retrieve the protobuf graph definition and fix the batch norm nodes
        gd = sess.graph.as_graph_def()
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # Get the list of important nodes
        output_node_names = 'res_prediction'
        whitelist_names = []
        for node in gd.node:
            # if node.name.startswith('InceptionResnetV1') or node.name.startswith('embeddings') or node.name.startswith('phase_train') or node.name.startswith('Bottleneck'):
            print(node.name)
            if not node.name.startswith('Logits'):
                whitelist_names.append(node.name)

        # Replace all the variables in the graph with constants of the same values
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, gd, output_node_names.split(","),
            variable_names_whitelist=whitelist_names)

        if True:
            model_save_dir = os.path.join(model_dir, 'model')
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            model_name = model_def.split('.')[1] + '-' + str(image_size)

            step = 0
            save_variables_and_metagraph(sess, saver, model_save_dir, model_name, step)
            graph_def = sess.graph.as_graph_def()

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print(model_path)
        print(output_file)


def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    import time
    # Save the model checkpoint
    print('Save variables ' + model_dir)
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Save metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)


if __name__ == '__main__':
    main()
