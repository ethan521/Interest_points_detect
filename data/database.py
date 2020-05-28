# coding=utf-8

import os
import numpy as np
import cv2
import sys
import copy
from face_data_augment import *
from common import annotate_bbox, annotate_points


class DataBaseDB:
    def __init__(self):
        self.data_name = "DataBaseDB"

    def get_next_test_batch_data(self):
        pass

    def get_next_train_batch_data(self):
        pass

    def get_next_train_batch_data(self):
        pass

    def img_preprocess_norm_128(self, img):
        img = (img - 128.) / 128.
        return img

    def img_preprocess_norm_256(self, img):
        img = img / 256.
        return img

    def img_attr_preprocess_norm(self, img):
        # return self.img_preprocess_norm_128(img)
        return self.img_preprocess_norm_256(img)


class AntiBaseDB(DataBaseDB):
    def __init__(self, image_size=96):
        super(AntiBaseDB, self).__init__()
        self.data_name = "AntiSpoofDB"
        self.image_size = image_size
        self.verbose = False
        self.is_training = True

    def get_antispoof_label_by_img_path(self, img_path):
        label = 1  # fake label
        if "CASIA_faceAntisp" in img_path:
            # 数据集: CASIA_faceAntisp
            video_name = os.path.basename(os.path.dirname(img_path))
            if video_name in [ "1", "2" ]:
                label = 0  # 真人
            elif video_name in [ "7", "8" ]:
                label = 1  # 电子屏幕
            else:
                label = 2  # 3，4，5，6打印纸张
            pass
        else:
            # 其他数据集:
            if 'live' in img_path:
                label = 0  # 真人
            elif ('fake_paper' in img_path
                  or 'print' in img_path):
                label = 2  # 打印纸张
            else:
                label = 1
        return label

    def img_augument_process(self, img, img_size, aug_proba=1.0):
        try:
            pass
            # 图片大小尺寸发生改变start#################################################################################
            # img = copy.copy(image)
            # if False:
            # 旋转
            if np.random.random() < aug_proba * 0.3:
                img = random_rotate(img, 30, True)
            # pad
            # if np.random.random() < aug_proba * 0.1:
            #     img = random_pad(img, 0.005, 0.15)
            # 随机裁剪
            if np.random.random() < aug_proba * 0.13:
                img = random_crop(img, 0.95, 0.15)
            # # 遮挡-裁剪
            if np.random.random() < aug_proba * 0.2:
                img = random_occlusion(img,area_ratio=0.08,hw_vari=0.6)

                # if img is None or len(img) == 0:
                #     img = image
        except Exception as e:
            print("图片大小尺寸发生改变end")
        try:
            # if (img.shape[ 0 ] != image_size) or (img.shape[ 1 ] != img_size):
            #     img = cv2.resize(img, (img_size, img_size))
            # 图片缩放
            img = cv2.resize(img, (img_size, img_size))
        except Exception as e:
            print('error-' * 10)
            print(type(img))
            print(img)
            print(e, "cv2.resize")
        try:
            # 图片大小尺寸发生改变end#######################################################################################
            # 图片细节增强
            if np.random.random() < aug_proba * 0.01:
                img = detail_enhance(img, np.random.random() * 4)
            # 边缘保持
            if np.random.random() < aug_proba * 0.003:
                img = edge_preserve(img, np.random.random() * 3)
            # 饱和度
            if np.random.random() < aug_proba * 0.1:
                img = change_saturation(img, -20 + np.random.random() * 40)
            # 亮度
            if np.random.random() < aug_proba:
                img = change_darker(img, -8 + np.random.random() * 16)
            # 颜色灰度化
            # if np.random.random() < aug_proba * 0.3:
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # 颜色翻转
            if np.random.random() < aug_proba * 0.3:
                img = 255 - img

            if np.random.random() < aug_proba * 1.3:
                #print(1111)
                img = cv2.flip(img,1)
            # 颜色通道互转
            if np.random.random() < aug_proba * 0.1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if self.verbose:
                # img = common.annotate_bbox(img, bbox)
                # img = common.annotate_points(img, points)
                cv2.imshow('test', img)
                cv2.waitKey(0)
                # print('data shape{}'.format(data.shape))
        except Exception as e:
            print(e, "图片细节增强")
        return img

    def preprocess_img_use_as_data(self, img,
                                   as_data=None,
                                   confidence=50,
                                   bbox_size=50,
                                   is_verbose=False,
                                   test_pad_scale=1.0):
        '''如果人脸关键点和人脸框辅助信息存在，则利用辅助信息检测；否则居中裁剪
        :param img:
        :param confidence: bbox assist data confidence(generate by mtcnn)
        :param as_data: bbox_x1,bbox_y1,bbox_x2,bbox_y2,score,points_x1,points_y1,....pints_x5,points_y5
        :return: 返回结果 data
        '''
        assert img is not None

        def img_center_crop(image: object, default_size: object = 240) -> object:
            '''
                如果图像尺寸大于default,居中剪裁default;
                如果图像小于240，则按照image较小尺寸的一边居中裁剪
            :param image:
            :return:
            '''
            # 居中剪裁压缩
            minsize = min((image.shape[ 0 ], image.shape[ 1 ], default_size))
            start_x = (image.shape[ 1 ] - minsize) // 2
            start_y = (image.shape[ 0 ] - minsize) // 2
            # data = image[ start_x:start_x + minsize, start_y:start_y + minsize ]
            # return data, True
            return image[ start_x:start_x + minsize, start_y:start_y + minsize ]

        if as_data is None:
            return img_center_crop(img)

        else:
            # 根据人脸鼻尖关键点，居中建材压缩
            as_label = np.array(as_data, dtype=int)
            if as_label[ 4 ] < confidence:
                # bbox置信度小于confidence，则居中裁剪
                # return img_center_crop(img)
                return None

            if is_verbose:
                points = np.zeros((5, 2))
                bbox = as_label[ : 4 ]
                bbox_h = bbox[ 3 ] - bbox[ 1 ]
                bbox_w = bbox[ 2 ] - bbox[ 0 ]
                points[ :, 0 ] = as_label[ 5::2 ]
                points[ :, 1 ] = as_label[ 6::2 ]
                img = annotate_points(img, points)
                img = annotate_bbox(img, bbox)  # x,y
                cv2.imshow("img", img)
                cv2.waitKey(0)

            # nose_x = points[ 2 ][ 0 ]
            # nose_y = points[ 2 ][ 1 ]
            nose_x = as_label[ 9 ]
            nose_y = as_label[ 10 ]
            bbox_h = as_label[ 3 ] - as_label[ 1 ]
            bbox_w = as_label[ 2 ] - as_label[ 0 ]

            if bbox_w > bbox_size \
                    and bbox_h > bbox_size \
                    and as_label[ 0 ] < nose_x < as_label[ 2 ] \
                    and as_label[ 1 ] < nose_y < as_label[ 3 ]:
                # 鼻尖在bbox范围内，bbox尺寸需要大于bbox_size
                if self.is_training:
                    # random pad
                    padsize = (0.9 * np.random.random() + 0.9) * bbox_h
                else:
                    padsize = bbox_h*test_pad_scale
                start_x = int(max(nose_x - padsize, 0))
                start_y = int(max(nose_y - 1.1 * padsize, 0))
                end_x = int(min(nose_x + padsize, img.shape[ 1 ]))
                end_y = int(min(nose_y + padsize, img.shape[ 0 ]))
                data = img[ start_y:end_y, start_x:end_x ]
                return data
            else:
                # return img_center_crop(img)
                return None
