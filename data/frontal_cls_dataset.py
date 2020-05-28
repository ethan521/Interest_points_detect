# coding=utf-8

import os
import numpy as np
import cv2
import sys
import copy
from sample_rgb_data_from_video.face_data_augment import *
# from utils_file import *
from common import scan_image_tree_get_relative_path, scan_folder_tree_get_relative_path, scan_image_tree
from common import annotate_bbox, annotate_points


from data.database import DataBaseDB


class FrontalFacefDB(DataBaseDB):
    def __init__(self, tmp_folder, data_batch_size=512,image_size=96):
        super(FrontalFacefDB, self).__init__()
        self.data_name = "FrontalFacefDB"
        self.image_size = image_size

        self.cls_folder = r'D:\data\face-attr\cls_face'  # 数据集路径
        self.cls_image_list_neg_folder = [
            'negative',
            'negative_hard',
            'negative_hard_angle',
            'negative_hard_angle_aug',
            'negative_aug',
            'negative_IR_scence',
            'negative_over_export_out_select',
            'negative_IR_scence_aug'
        ]
        self.cls_image_list_part_folder = [
            'part',
            'part_aug',
            'part_head',
            'part_remove_yaw',
            'part_remove_yaw_aug',
            'part_generate',
            'part_generate_aug'
        ]
        self.cls_image_list_pos_folder = [
            # 'positive',
            #  'positive_1',
            #  'positive_2',
            #  'positive_pub',
            #  'positive_pub_close',
            'positive_pub_cloud',
            'positive_NUAA_clients_face',
            'positive_NUAA_imposter_face',
            'positive_nir_aligned'
        ]
        self.cls_image_list_test_folder = [
            'positive_test_6',
            'test_negative_aug',
            'negative_hard' ]


        self.cls_image_list_neg = None
        self.cls_image_list_neg_num = 0
        self.cls_image_list_neg_index = 0

        self.cls_image_list_part = None
        self.cls_image_list_part_num = 0
        self.cls_image_list_part_index = 0

        self.cls_image_list_pos = None
        self.cls_image_list_pos_num = 0
        self.cls_image_list_pos_index = 0

        self.cls_image_list_test = None  # 测试集
        self.cls_image_list_test_num = 0
        self.cls_image_list_test_index = 0

        self.test_bach_size = data_batch_size
        self.train_bach_size = data_batch_size
        self.train_bach_size_neg = data_batch_size // 3
        self.train_bach_size_part = data_batch_size // 3
        self.train_bach_size_pos = self.train_bach_size - self.train_bach_size_neg - self.train_bach_size_part

        self.verbose = False
        self.aug_proba = 1.0

        self.load_data_file_list(tmp_folder)

        self.permutation_test_dataset()
        self.permutation_train_dataset()

        print('-' * 50)
        print('init {} done'.format(self.data_name))
        print('test num:{}'.format(self.cls_image_list_neg_num))
        print('train num:{}'.format(
            self.cls_image_list_part_num + self.cls_image_list_pos_num + self.cls_image_list_neg_num))
        print('train negtive num:{}'.format(self.cls_image_list_neg_num))
        print('train part num:{}'.format(self.cls_image_list_part_num))
        print('train positive num:{}'.format(self.cls_image_list_pos_num))
        print('-' * 50)
        pass

    def load_data_file_list(self, tmp_folder, is_reload=True):

        path_neg = 'cls_image_list_neg.txt'
        path_part = 'cls_image_list_part.txt'
        path_pos = 'cls_image_list_pos.txt'
        path_test = 'cls_image_list_test.txt'

        if os.path.exists(tmp_folder) \
                and path_neg in os.listdir(tmp_folder) \
                and path_part in os.listdir(tmp_folder) \
                and path_pos in os.listdir(tmp_folder) \
                and path_test in os.listdir(tmp_folder)\
                and is_reload:

            with open(os.path.join(tmp_folder, path_test), 'r') as f:
                cont = [ e.strip() for e in f.readlines() ]
                self.cls_image_list_test = np.array(cont)
                self.cls_image_list_test_num = len(cont)

            with open(os.path.join(tmp_folder, path_neg), 'r') as f:
                cont = [ e.strip() for e in f.readlines() ]
                self.cls_image_list_neg = np.array(cont)
                self.cls_image_list_neg_num = len(cont)

            with open(os.path.join(tmp_folder, path_part), 'r') as f:
                cont = [ e.strip() for e in f.readlines() ]
                self.cls_image_list_part = np.array(cont)
                self.cls_image_list_part_num = len(cont)

            with open(os.path.join(tmp_folder, path_pos), 'r') as f:
                cont = [ e.strip() for e in f.readlines() ]
                self.cls_image_list_pos = np.array(cont)
                self.cls_image_list_pos_num = len(cont)

        else:
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)
            ########## add cls data
            max_count = 400000
            cls_image_list_test = [ ]
            for name in self.cls_image_list_test_folder:
                raw_list = np.array(os.listdir(os.path.join(self.cls_folder, name)))
                raw_list = raw_list[ np.random.permutation(np.arange(0, len(raw_list))) ][ :max_count ]
                cls_image_list_test += [ os.path.join(self.cls_folder, name, e) for e in raw_list ]
            self.cls_image_list_test = np.array(cls_image_list_test)
            self.cls_image_list_test_num = len(cls_image_list_test)

            np.savetxt(os.path.join(tmp_folder, path_test), self.cls_image_list_test, fmt='%s')

            cls_image_list_neg = [ ]
            max_count = 100000
            for name in self.cls_image_list_neg_folder:
                raw_list = np.array(os.listdir(os.path.join(self.cls_folder, name)))
                raw_list = raw_list[ np.random.permutation(np.arange(0, len(raw_list))) ][ :max_count ]
                print('{} len:{}'.format(name, len(raw_list)))
                cls_image_list_neg += [ os.path.join(self.cls_folder, name, e) for e in raw_list ]

            max_count = 100000
            cls_image_list_part = [ ]
            # for name in ['part', 'part_remove_yaw', 'part_generate']:
            for name in self.cls_image_list_part_folder:
                raw_list = np.array(os.listdir(os.path.join(self.cls_folder, name)))
                raw_list = raw_list[ np.random.permutation(np.arange(0, len(raw_list))) ][ :max_count ]
                print('{} len:{}'.format(name, len(raw_list)))
                cls_image_list_part += [ os.path.join(self.cls_folder, name, e) for e in raw_list ]

            cls_image_list_pos = [ ]
            max_count = 4000000
            for name in self.cls_image_list_pos_folder:
                if name in [ 'positive_nir_aligned' ]:  # 递归
                    raw_list = [ ]
                    self.get_images(os.path.join(self.cls_folder, name), raw_list)
                    # raw_list = np.array(os.listdir(os.path.join(cls_folder, name)))
                    raw_list = np.array(raw_list)
                    raw_list = raw_list[ np.random.permutation(np.arange(0, len(raw_list))) ][ :max_count ]
                    print('{} len:{}'.format(name, len(raw_list)))
                else:
                    raw_list = np.array(os.listdir(os.path.join(self.cls_folder, name)))
                    raw_list = raw_list[ np.random.permutation(np.arange(0, len(raw_list))) ][ :max_count ]
                    print('{} len:{}'.format(name, len(raw_list)))
                cls_image_list_pos += [ os.path.join(self.cls_folder, name, e) for e in raw_list ]

            if len(cls_image_list_neg) is None or len(cls_image_list_part) is None or len(cls_image_list_pos) is None:
                pass

            print('cls_image_list_neg:{}'.format(len(cls_image_list_neg)))
            print('cls_image_list_part:{}'.format(len(cls_image_list_part)))
            print('cls_image_list_pos:{}'.format(len(cls_image_list_pos)))
            self.cls_image_list_neg = np.array(cls_image_list_neg)
            self.cls_image_list_neg_num = len(cls_image_list_neg)
            self.cls_image_list_part = np.array(cls_image_list_part)
            self.cls_image_list_part_num = len(cls_image_list_part)
            self.cls_image_list_pos = np.array(cls_image_list_pos)
            self.cls_image_list_pos_num = len(cls_image_list_pos)

            # save
            np.savetxt(os.path.join(tmp_folder, path_neg), self.cls_image_list_neg, fmt='%s')
            np.savetxt(os.path.join(tmp_folder, path_part), self.cls_image_list_part, fmt='%s')
            np.savetxt(os.path.join(tmp_folder, path_pos), self.cls_image_list_pos, fmt='%s')

            print('save cls_image_list_test.txt done.')

    def get_images(self, folder, image_path_list):
        ''' 递归查找所有图片文件
        :param folder: 输入根目录
        :param image_path_list: 缓存结果
        :return:
        '''
        list_temp = os.listdir(folder)
        if len(list_temp) > 0:
            path = os.path.join(folder, list_temp[ 0 ])
            if os.path.isfile(path):
                image_path_list += [ os.path.join(folder, e) for e in list_temp ]
            else:
                for temp in list_temp:
                    next_folder = os.path.join(folder, temp)
                    self.get_images(next_folder, image_path_list)

    def get_next_test_batch_data(self):
        if np.random.random() < 0.05:
            self.permutation_test_dataset()

        data, label, index = self.get_image_and_label(self.test_bach_size,
                                                      self.cls_image_list_test,
                                                      self.cls_image_list_test_num,
                                                      self.cls_image_list_test_index,
                                                      is_training=False)
        self.cls_image_list_test_index = index
        return data, label

    def get_next_train_batch_data(self):
        if np.random.random() < 0.03:
            self.permutation_train_dataset()

        batch_data, batch_label = [ ], [ ]
        data, label, index = self.get_image_and_label(self.train_bach_size_neg,
                                                      self.cls_image_list_neg,
                                                      self.cls_image_list_neg_num,
                                                      self.cls_image_list_neg_index,
                                                      is_training=True)
        self.cls_image_list_neg_index = index
        batch_data += data
        batch_label += label

        data, label, index = self.get_image_and_label(self.train_bach_size_part,
                                                      self.cls_image_list_part,
                                                      self.cls_image_list_part_num,
                                                      self.cls_image_list_part_index,
                                                      is_training=True)
        self.cls_image_list_part_index = index
        batch_data += data
        batch_label += label

        data, label, index = self.get_image_and_label(self.train_bach_size_pos,
                                                      self.cls_image_list_pos,
                                                      self.cls_image_list_pos_num,
                                                      self.cls_image_list_pos_index,
                                                      is_training=True)
        self.cls_image_list_pos_index = index
        batch_data += data
        batch_label += label
        return np.array(batch_data), np.array(batch_label)

    def permutation_train_dataset(self):
        # random
        permu_index = np.random.permutation(np.arange(0, self.cls_image_list_neg_num))
        self.cls_image_list_neg = self.cls_image_list_neg[ permu_index ]

        permu_index = np.random.permutation(np.arange(0, self.cls_image_list_part_num))
        self.cls_image_list_part = self.cls_image_list_part[ permu_index ]

        permu_index = np.random.permutation(np.arange(0, self.cls_image_list_pos_num))
        self.cls_image_list_pos = self.cls_image_list_pos[ permu_index ]

    def permutation_test_dataset(self):
        # random
        permu_index = np.random.permutation(np.arange(0, self.cls_image_list_test_num))
        self.cls_image_list_test = self.cls_image_list_test[ permu_index ]

    def get_frontal_cls_label_by_img_path(self, label_path):
        label = 1
        if 'positive' in label_path:
            label = 1
        elif 'part' in label_path:
            label = 0
        elif 'negative' in label_path:
            label = 0
        return label

    def get_image_and_label(self,
                            sub_batch_size,
                            file_path_list,
                            num_data,
                            img_index=0,
                            is_training=False):
        '''
        :param sub_batch_size:  批次大小
        :param file_path_list:  数据集图片list
        :param num_data:    数据集大小
        :param img_index:   图片索引
        :param is_training:
        :return:
        '''

        img_list, cls_label_list = [ ], [ ]
        count = 0
        while count < sub_batch_size:
            i = img_index
            img_index += 1
            if img_index == num_data - 1:
                img_index = 0

            path = file_path_list[ i ]
            label = self.get_frontal_cls_label_by_img_path(path)
            img = cv2.imread(path)

            if img is None:
                continue
            try:
                # # 不需要辅助信息--正脸检测时候
                # # 利用辅助信息
                # as_label_path = path + '.txt'
                # if os.path.exists(as_label_path):
                #     # 利用人脸关键点辅助信息裁剪
                #     as_data = np.loadtxt(as_label_path, dtype=int)
                #     data, flag = self.preprocess_img_use_as_data(img, as_data=as_data)
                #     if flag:
                #         img = data
                #     else:
                #         img, _ = self.preprocess_img_use_as_data(img)
                # else:
                #     if img.shape[ 0 ] > image_size + 50:
                #         img, _ = self.preprocess_img_use_as_data(img)
                #
                # # 数据增广处理
                # if img is None:
                #     continue

                # print(label)
                img = self.img_augument_process(img, self.image_size, aug_proba=1.0, verbose=self.verbose)
                # 图片前置处理 归一化..
                img = self.img_attr_preprocess_norm(img)
                img_list += [ img ]
                cls_label_list += [ label ]
                count += 1

            except Exception as e:
                print(e, path)
        return img_list, cls_label_list, img_index

    def img_augument_process(self, image, img_size, aug_proba=1.0, verbose=False):

        # 图片大小尺寸发生改变end  #######################################################################################
        img = copy.copy(image)
        # # 旋转
        # if np.random.random() < aug_proba * 0.3:
        #     img = random_rotate(img, 30, True)
        # # pad
        # if np.random.random() < aug_proba * 0.1:
        #     img = random_pad(img, 0.005, 0.15)
        # # 随机裁剪
        # if np.random.random() < aug_proba * 0.3:
        #     img = random_crop(img, 0.95, 0.15)
        # # # 遮挡-裁剪
        # if np.random.random() < aug_proba * 0.2:
        #     img = random_occlusion(img)
        # if img is None or len(img) == 0:
        #     img = image
        # # if (img.shape[ 0 ] != image_size) or (img.shape[ 1 ] != img_size):
        # #     img = cv2.resize(img, (img_size, img_size))
        # 图片缩放
        img = cv2.resize(img, (img_size, img_size))
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
        if np.random.random() < aug_proba * 0.3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # 颜色翻转
        if np.random.random() < aug_proba * 0.3:
            img = 255 - img
        # 颜色通道互转
        if np.random.random() < aug_proba * 0.1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if verbose:
            cv2.imshow('test', img)
            cv2.waitKey(0)
            # print('data shape{}'.format(data.shape))
        return img


if __name__ == '__main__':
    tmp_folder = r'data_train_cls_temp'
    batch_size = 100
    db = FrontalFacefDB(tmp_folder, batch_size)
    db.verbose = True
    for i in range(100):
        db.get_next_test_batch_data()
        db.get_next_train_batch_data()
    pass
