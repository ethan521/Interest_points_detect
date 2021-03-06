"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def l2_regularizer(scale, scope=None):
    def l2(weights):
        """Applies l2 regularization to weights."""
        weights = tf.cast(weights, tf.float32)
        from tensorflow.python.ops import nn
        from tensorflow.python.ops import standard_ops
        from tensorflow.python.framework import ops
        with ops.name_scope(scope, 'l2_regularizer', [weights]) as name:
            my_scale = ops.convert_to_tensor(scale,
                                             dtype=weights.dtype.base_dtype,
                                             name='scale')
            return standard_ops.multiply(my_scale, nn.l2_loss(weights), name=name)
    return l2

def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return irv1_small2(images, is_training=phase_train,
                           dropout_keep_prob=keep_probability, reuse=reuse)


def irv1_small2(inputs, is_training=True,
                dropout_keep_prob=0.8,
                reuse=None,
                scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
    keys_list =[]

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):

                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='SAME',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                keys_list += ['Conv2d_1a_3x3']
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='SAME',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                keys_list += ['Conv2d_2a_3x3']
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                keys_list += ['Conv2d_2b_3x3']
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                keys_list += ['MaxPool_3a_3x3']
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='SAME',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                keys_list += ['Conv2d_3b_1x1']
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='SAME',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                keys_list += ['Conv2d_4a_3x3']
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='SAME',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                keys_list += ['Conv2d_4b_3x3']

                # 35/2 x 35/2 x 512
                net = slim.conv2d(net, 512, 3, stride=2, padding='SAME',
                                  scope='Conv2d_5a_3x3')
                end_points['Conv2d_5a_3x3'] = net
                keys_list += ['Conv2d_5a_3x3']

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    keys_list += ['PrePool']
                    #pylint: disable=no-member
                    print(net.get_shape())

                    with tf.variable_scope(scope, 'MultiScaleLayer', [inputs], reuse=reuse):
                        def get_tensors(net_temp):
                            net_temp = slim.avg_pool2d(net_temp, net_temp.get_shape()[1:3], padding='VALID',
                                                       scope='AvgPool')
                            net_temp = slim.flatten(net_temp)
                            return net_temp

                        net0 = get_tensors(end_points['Conv2d_5a_3x3'])
                        net1 = get_tensors(end_points['Conv2d_4b_3x3'])
                        net2 = get_tensors(end_points['Conv2d_4a_3x3'])
                        net = tf.concat([net0, net1, net2], 1)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

                    end_points['PreLogitsFlatten'] = net
                    keys_list += ['PreLogitsFlatten']

                for k in keys_list:
                    print(k, end_points[k].get_shape(), end_points[k].dtype)

    return net, end_points

