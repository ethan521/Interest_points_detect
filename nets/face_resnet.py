import sys
# sys.path.insert(0, '..')
from network_caffe import Network

class FaceResNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1a')
             .prelu(name='relu1a')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv1b')
             .prelu(name='relu1b')
             .max_pool(2, 2, 2, 2, name='pool1b')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv2_1')
             .prelu(name='relu2_1')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv2_2')
             .prelu(name='relu2_2'))

        (self.feed('pool1b', 
                   'relu2_2')
             .add(name='res2_2')
             .conv(3, 3, 128, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='relu2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv3_1')
             .prelu(name='relu3_1')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv3_2')
             .prelu(name='relu3_2'))

        (self.feed('pool2', 
                   'relu3_2')
             .add(name='res3_2')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv3_3')
             .prelu(name='relu3_3')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv3_4')
             .prelu(name='relu3_4'))

        (self.feed('res3_2', 
                   'relu3_4')
             .add(name='res3_4')
             .conv(3, 3, 256, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='relu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_1')
             .prelu(name='relu4_1')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_2')
             .prelu(name='relu4_2'))

        (self.feed('pool3', 
                   'relu4_2')
             .add(name='res4_2')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_3')
             .prelu(name='relu4_3')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_4')
             .prelu(name='relu4_4'))

        (self.feed('res4_2', 
                   'relu4_4')
             .add(name='res4_4')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_5')
             .prelu(name='relu4_5')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_6')
             .prelu(name='relu4_6'))

        (self.feed('res4_4', 
                   'relu4_6')
             .add(name='res4_6')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_7')
             .prelu(name='relu4_7')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_8')
             .prelu(name='relu4_8'))

        (self.feed('res4_6', 
                   'relu4_8')
             .add(name='res4_8')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_9')
             .prelu(name='relu4_9')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4_10')
             .prelu(name='relu4_10'))

        (self.feed('res4_8', 
                   'relu4_10')
             .add(name='res4_10')
             .conv(3, 3, 512, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='relu4')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_1')
             .prelu(name='relu5_1')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_2')
             .prelu(name='relu5_2'))

        (self.feed('pool4', 
                   'relu5_2')
             .add(name='res5_2')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_3')
             .prelu(name='relu5_3')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_4')
             .prelu(name='relu5_4'))

        (self.feed('res5_2', 
                   'relu5_4')
             .add(name='res5_4')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_5')
             .prelu(name='relu5_5')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_6')
             .prelu(name='relu5_6'))

        (self.feed('res5_4', 
                   'relu5_6')
             .add(name='res5_6')
             .fc(128, relu=False, name='fc5'))

