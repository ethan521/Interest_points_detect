from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models.network as network

def inception(inp, inSize, ks, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2, o4s3, poolType, name, 
              phase_train=True, use_batch_norm=True, weight_decay=0.0):
  
    print('name = ', name)
    print('inputSize = ', inSize)
    print('kernelSize = {3,5}')
    print('kernelStride = {%d,%d}' % (ks,ks))
    print('outputSize = {%d,%d}' % (o2s2,o3s2))
    print('reduceSize = {%d,%d,%d,%d}' % (o2s1,o3s1,o4s2,o1s))
    print('pooling = {%s, %d, %d, %d, %d}' % (poolType, o4s1, o4s1, o4s3, o4s3))
    if (o4s2>0):
        o4 = o4s2
    else:
        o4 = inSize
    print('outputSize = ', o1s+o2s2+o3s2+o4)
    print()
    
    net = []

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      with tf.variable_scope(name):
        with tf.variable_scope('branch1_1x1'):
          if o1s > 0:
            branch_1 = slim.conv2d(net, depth(o1s), [1, 1], scope='conv1x1')
        with tf.variable_scope('branch2_3x3'):
          if o2s1 > 0:
            branch_2 = slim.conv2d(net, depth(o2s1), [1, 1], scope='conv1x1')
            branch_2 = slim.conv2d(branch_2, depth(o2s1), [3, 3], stride=ks, scope='conv3x3')
        with tf.variable_scope('branch3_5x5'):
          if o3s1 > 0:
            branch_3 = slim.conv2d(net, depth(o3s1), [1, 1], scope='conv1x1')
            branch_3 = slim.conv2d(branch_3, depth(o3s1), [5, 5], stride=ks, scope='conv3x3')

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                 scope='Conv2d_0b_1x1')

        with tf.variable_scope('branch4_pool'):
            if poolType=='MAX':
                pool = mpool(inp, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            elif poolType=='L2':
                pool = lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            else:
                raise ValueError('Invalid pooling type "%s"' % poolType)
            
            if o4s2>0:
                branch_4 = slim.conv2d(net, depth(o4s2), [1, 1], scope='conv1x1')
            else:
                branch_4 = pool

        net = tf.concat(axis=3, values=[branch_1, branch_2, branch_3, branch_4])
    
    with tf.variable_scope(name):
        with tf.variable_scope('branch1_1x1'):
            if o1s>0:
                conv1 = conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv1)
      
        with tf.variable_scope('branch2_3x3'):
            if o2s1>0:
                conv3a = conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv3 = conv(conv3a, o2s1, o2s2, 3, 3, ks, ks, 'SAME', 'conv3x3', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv3)
      
        with tf.variable_scope('branch3_5x5'):
            if o3s1>0:
                conv5a = conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv5 = conv(conv5a, o3s1, o3s2, 5, 5, ks, ks, 'SAME', 'conv5x5', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv5)
      
        with tf.variable_scope('branch4_pool'):
            if poolType=='MAX':
                pool = mpool(inp, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            elif poolType=='L2':
                pool = lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            else:
                raise ValueError('Invalid pooling type "%s"' % poolType)
            
            if o4s2>0:
                pool_conv = conv(pool, inSize, o4s2, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
            else:
                pool_conv = pool
            net.append(pool_conv)
      
        incept = array_ops.concat(net, 3, name=name)
    return incept



def inference(images, keep_probability, phase_train=True, weight_decay=0.0):
    """ Define an inference network for face recognition based 
           on inception modules using batch normalization
    
    Args:
      images: The images to run inference on, dimensions batch_size x height x width x channels
      phase_train: True if batch normalization should operate in training mode
    """
    endpoints = {}
    net = network.conv(images, 3, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv1'] = net
    net = network.mpool(net,  3, 3, 2, 2, 'SAME', 'pool1')
    endpoints['pool1'] = net
    net = network.conv(net,  64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv2_1x1'] = net
    net = network.conv(net,  64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv3_3x3'] = net
    net = network.mpool(net,  3, 3, 2, 2, 'SAME', 'pool3')
    endpoints['pool3'] = net
  
    net = network.inception(net, 192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'MAX', 'incept3a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3a'] = net
    net = network.inception(net, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 'MAX', 'incept3b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3b'] = net
    net = network.inception(net, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'MAX', 'incept3c', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3c'] = net
    
    net = network.inception(net, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, 'MAX', 'incept4a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4a'] = net
    net = network.inception(net, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1, 'MAX', 'incept4b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4b'] = net
    net = network.inception(net, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1, 'MAX', 'incept4c', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4c'] = net
    net = network.inception(net, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1, 'MAX', 'incept4d', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4d'] = net
    net = network.inception(net, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'MAX', 'incept4e', phase_train=phase_train, use_batch_norm=True)
    endpoints['incept4e'] = net
    
    net = network.inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1, 'MAX', 'incept5a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5a'] = net
    net = network.inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1, 'MAX', 'incept5b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5b'] = net
    net = network.apool(net,  5, 5, 1, 1, 'VALID', 'pool6')
    endpoints['pool6'] = net
    net = tf.reshape(net, [-1, 1024])
    endpoints['prelogits'] = net
    net = tf.nn.dropout(net, keep_probability)
    endpoints['dropout'] = net
    
    return net, endpoints
