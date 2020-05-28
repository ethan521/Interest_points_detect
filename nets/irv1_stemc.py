from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def prelu(input):
    i = input.get_shape().as_list()
    alpha = tf.get_variable('alpha', i[-1],
                            initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(input)
    neg = -alpha * tf.nn.relu(-input)
    output = pos + neg
    return output


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
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=prelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return irv1_stemc(images, is_training=phase_train,
                          dropout_keep_prob=keep_probability, reuse=reuse)


def irv1_stemc(inputs, is_training=True,
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

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):

                print('inputs ' + str(inputs.get_shape()))
                with tf.variable_scope('irv1_crelu'):
                    activation_fn = tf.nn.crelu
                    # 149 x 149 x 32
                    net = slim.conv2d(inputs, 32/2, 3, stride=2, padding='VALID',
                                      activation_fn=activation_fn, scope='Conv2d_1a_3x3')
                    end_points['Conv2d_1a_3x3'] = net
                    print('Conv2d_1a_3x3 ' + str(net.get_shape()))
                    # 147 x 147 x 32
                    net = slim.conv2d(net, 32/2, 3, stride=2, padding='VALID',
                                      activation_fn=activation_fn, scope='Conv2d_2a_3x3')
                    end_points['Conv2d_2a_3x3'] = net
                    print('Conv2d_2a_3x3 ' + str(net.get_shape()))
                    # 147 x 147 x 64
                    net = slim.conv2d(net, 64/2, 3, activation_fn=activation_fn, scope='Conv2d_2b_3x3')
                    end_points['Conv2d_2b_3x3'] = net
                    print('Conv2d_2b_3x3 ' + str(net.get_shape()))

                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                print('MaxPool_3a_3x3 ' + str(net.get_shape()))
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                print('Conv2d_3b_1x1 ' + str(net.get_shape()))
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                print('Conv2d_4a_3x3 ' + str(net.get_shape()))
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                print('Conv2d_4b_3x3 ' + str(net.get_shape()))

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

    return net, end_points

