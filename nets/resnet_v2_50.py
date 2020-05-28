# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

from models import resnet_utils
from models.resnet_v2 import resnet_v2_50

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):
    with slim.arg_scope(resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v2_50(images,
                     num_classes=None,
                     is_training=phase_train,
                     global_pool=True,
                     output_stride=None,
                     spatial_squeeze=True,
                     reuse=reuse,
                     scope='resnet_v2_50')
        end_points['PreLogitsFlatten'] = slim.flatten(end_points['global_pool'])
        resnet_v2_50.default_image_size = images.get_shape()[1]
        return end_points['PreLogitsFlatten'], end_points

