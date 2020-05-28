# coding=utf-8

import os
import numpy as np
import cv2
import pickle
import sys
import copy
from face_data_augment import *
# from utils_file import *
from common import scan_image_tree_get_relative_path, scan_folder_tree_get_relative_path, scan_image_tree
from common import annotate_bbox, annotate_points

image_size = 96

from data.database import DataBaseDB
import common
import json
import cv2
# from  face_data_augment import random_bbox, image_augment_cv


class AttrFacefDB(DataBaseDB):
    def __init__(self, tmp_folder, data_batch_size=512):
        '''
        :param tmp_folder:
        :param data_batch_size:

        init AttrFacefDB done
        test num:20119
        train num:216253
        '''
        super(AttrFacefDB, self).__init__()
        self.data_name = "AttrFacefDB"
        self.image_size = image_size
        self.train_dir = r"..\running_tmp\20200228-integrate-96x96-cls-cls4-attr-antispoof"
        self.dataset_folder = r'D:\data\face-attr\louyu-capture'
        self.train_image_dir_list = [
            # r'D:\data\face-attr\celeba-2w',
            r'D:\data\face-attr\louyu-capture\downloadimg-20180905',
            r'D:\data\face-attr\louyu-capture\downloadimg-rec-20181025',
            r'D:\data\face-attr\louyu-capture\downloadimg-rec-20181127',
            r'D:\data\face-attr\louyu-capture\downloadimg-rec-20181228',
            #
            r'D:\data\face-attr\attr-with-pts68\cacd2000-young-old\cacd2000-young-old',
            r'D:\data\face-attr\attr-with-pts68\semifrontal\semifrontal'
        ]

        self.test_image_dir_list = [
            r'D:\data\face-attr\louyu-capture\downloadimg-20180911',
            # r'E:\data\face-attr\louyu-capture\louyu-7ch-night-20180123'
        ]

        self.train_images_list = None
        self.train_labels_dict = None
        self.train_images_list_num = 0
        self.train_images_list_index = 0

        self.test_images_list = None
        self.test_labels_dict = None
        self.test_images_list_num = 0
        self.test_images_list_index = 0

        self.test_bach_size = data_batch_size
        self.train_bach_size = data_batch_size

        self.verbose = False
        self.aug_proba = 1.0

        self.load_data_file_list(tmp_folder)
        self.is_select_hard_sample = False

        if self.is_select_hard_sample:
            self.train_images_list_bk = self.train_images_list
            self.train_labels_dict_bk = self.train_labels_dict
            self.train_images_list_num_bk = self.train_images_list_num
            self.train_images_list_index_bk = self.train_images_list_index

        self.permutation_test_dataset()
        self.permutation_train_dataset()

        print('-' * 50)
        print('init {} done'.format(self.data_name))
        print('test num:{}'.format(self.test_images_list_num))
        print('train num:{}'.format(self.train_images_list_num))
        print('-' * 50)
        pass

    def load_data_file_list(self, tmp_folder, is_reload=True):

        train_file_list_txt = os.path.join(tmp_folder, "train_file_list.txt")
        train_labels_dict_pickle = os.path.join(tmp_folder, "train_labels_dict.pickle")
        test_file_list_txt = os.path.join(tmp_folder, "test_file_list.txt")
        test_labels_dict_pickle = os.path.join(tmp_folder, "test_labels_dict.pickle")
        if os.path.exists(tmp_folder) \
                and os.path.exists(train_file_list_txt) \
                and os.path.exists(train_labels_dict_pickle) \
                and os.path.exists(test_file_list_txt) \
                and os.path.exists(test_labels_dict_pickle) \
                and is_reload:

            # train_file_list = np.loadtxt(train_file_list_txt, dtype=str)
            with open(train_file_list_txt, 'r') as f:
                train_file_list = [ e.strip() for e in f.readlines() ]

            with open(train_labels_dict_pickle, 'rb') as handle:
                train_labels_dict = pickle.load(handle)

            # test_file_list = np.loadtxt(test_file_list_txt, dtype=str)
            with open(test_file_list_txt, 'r') as f:
                test_file_list = [ e.strip() for e in f.readlines() ]

            with open(test_labels_dict_pickle, 'rb') as handle:
                test_labels_dict = pickle.load(handle)

            self.test_images_list = np.array(test_file_list)
            self.test_images_list_num = len(test_file_list)

            self.test_labels_dict = test_labels_dict

            self.train_images_list = np.array(train_file_list)
            self.train_images_list_num = len(train_file_list)

            self.train_labels_dict = train_labels_dict

            pass
        else:
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)
            train_file_list = [ ]
            for image_dir in self.train_image_dir_list:
                print(image_dir)
                file_list = scan_image_tree(os.path.join(self.dataset_folder, image_dir))
                np.random.shuffle(file_list)
                # file_list = file_list[ :1500 ]
                # file_list = file_list[ :80000 ]
                train_file_list += file_list

            test_file_list = [ ]
            for image_dir in self.test_image_dir_list:
                print(image_dir)
                file_list = scan_image_tree(image_dir)  # 20203
                # file_list = file_list[:8000]
                # file_list = file_list[ :int(50 * self.test_bach_size + 200) ]
                test_file_list += file_list

            train_file_list, train_labels_dict, test_file_list, test_labels_dict = self.transform_file_list(
                train_file_list, test_file_list, verbose=0)

            self.test_images_list = np.array(test_file_list)
            self.test_labels_dict = test_labels_dict
            self.test_images_list_num = len(test_file_list)

            self.train_images_list = np.array(train_file_list)
            self.train_labels_dict = train_labels_dict
            self.train_images_list_num = len(train_file_list)

            np.savetxt(train_file_list_txt, train_file_list, fmt='%s')
            with open(train_labels_dict_pickle, 'wb') as handle:
                pickle.dump(train_labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            np.savetxt(test_file_list_txt, test_file_list, fmt='%s')
            with open(test_labels_dict_pickle, 'wb') as handle:
                pickle.dump(test_labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('save train and test cache data')
        pass

    def load_pts_label(self, filename):
        filename_pts = os.path.splitext(filename)[ 0 ] + '.pts'
        if not os.path.exists(filename_pts):
            return None
        with open(filename_pts) as f:
            gt_pts = [ ]
            lines = f.readlines()
            for line in lines[ 3:3 + 68 ]:
                l = line.strip().split(' ')
                gt_pts += [ [ float(l[ 1 ]), float(l[ 0 ]) ] ]
            return np.array(gt_pts)

    def cvt_68pts_to_5pts(self, points68):
        points5 = np.array([
            np.mean(points68[ 36 + 36::2 ][ :6 ], axis=0),  # left eye
            np.mean(points68[ 36 + 36 + 1::2 ][ :6 ], axis=0),  # left eye
            np.mean(points68[ 42 + 42::2 ][ :6 ], axis=0),  # right eye
            np.mean(points68[ 42 + 42 + 1::2 ][ :6 ], axis=0),  # right eye
            np.mean(points68[ 30 + 30::2 ][ :5 ], axis=0),  # node
            np.mean(points68[ 30 + 30 + 1::2 ][ :5 ], axis=0),  # node
            points68[ 48 + 48 ],  # left mouth
            points68[ 48 + 48 + 1 ],  # left mouth
            points68[ 54 + 54 ],  # right mouth
            points68[ 54 + 54 + 1 ] ]  # right mouth
        )
        # points5 = np.array([
        #    np.mean(points68[36:42], axis=0),  # left eye
        #    np.mean(points68[36:42], axis=0),  # left eye
        #    np.mean(points68[42:48], axis=0),  # right eye
        #    np.mean(points68[42:48], axis=0),  # right eye
        #    np.mean(points68[30:35], axis=0),  # node
        #   np.mean(points6#8[30:35], axis=0),  # node
        #    points68[48],  # left mouth
        #    points68[48],  # left mouth
        #    points68[54]],  # right mouth
        #    points68[54]]  # right mouth
        # )
        return points5

    def get_data_label(self, file_path_list):
        blur_list = [ ]
        pts68_list = [ ]
        idx_list = [ ]
        angle_list = [ ]
        age_list = [ ]
        beauty_list = [ ]
        expression_list = [ ]
        race_list = [ ]
        pts4_list = [ ]
        pts5_list = [ ]
        glasses_list = [ ]
        gender_list = [ ]
        for i, path in enumerate(file_path_list):
            sys.stdout.flush()
            sys.stdout.write('\r %d / %d' % (i, len(file_path_list)))

            img = cv2.imread(path)
            if img is None or img.shape[ 0 ] <= 0:
                print(path)
                continue
            path_face_attr = path[ :-4 ] + '.detect.json'
            if not os.path.exists(path_face_attr):
                continue

            import json
            with open(path_face_attr) as f:
                face_attr_dict = json.load(f)
                if 'result_num' not in face_attr_dict or face_attr_dict[ 'result_num' ] == 0:
                    continue
                result = face_attr_dict[ 'result' ][ 0 ]
                blur = result[ 'qualities' ][ 'blur' ]
                pitch = result[ 'pitch' ]
                yaw = result[ 'yaw' ]
                roll = result[ 'roll' ]

            pts68 = self.load_pts_label(path)
            if pts68 is None:
                continue
            pts68_ = pts68.ravel() / image_size
            pts68_list += [ pts68_ ]
            blur_list += [ blur ]
            idx_list += [ i ]
        sys.stdout.write('\n')
        labels_dict = {}
        labels_dict[ 'blur' ] = np.array(blur_list)
        labels_dict[ 'pts68' ] = np.array(pts68_list)
        labels_dict[ 'angle' ] = np.array(angle_list)
        labels_dict[ 'age' ] = np.array(age_list)
        labels_dict[ 'beauty' ] = np.array(beauty_list)
        labels_dict[ 'expression' ] = np.array(expression_list)
        labels_dict[ 'race' ] = np.array(race_list)
        labels_dict[ 'pts4' ] = np.array(pts4_list)
        labels_dict[ 'pts5' ] = np.array(pts5_list)
        labels_dict[ 'glasses' ] = np.array(glasses_list)
        labels_dict[ 'gender' ] = np.array(gender_list)
        file_list = file_path_list[ idx_list ]
        labels_dict = self.transform_labels_list(labels_dict, verbose=1)
        return file_list, labels_dict

    def transform_labels_list(self, labels_dict, verbose=0):
        blur_threshold = 0.05
        pos_idxes = np.where(labels_dict[ 'blur' ] > 0.8)[ 0 ]
        neg_idxes = np.where(labels_dict[ 'blur' ] <= blur_threshold)[ 0 ]
        labels_dict[ 'blur_pos_idxes' ] = pos_idxes
        labels_dict[ 'blur_neg_idxes' ] = neg_idxes
        if verbose > 0:
            print('blur', 'pos-neg', len(pos_idxes), len(neg_idxes))
        labels_dict[ 'blur' ] = (labels_dict[ 'blur' ] > blur_threshold).astype(int)
        # age
        cond = np.bitwise_or(labels_dict[ 'age' ] < 20, labels_dict[ 'age' ] > 50)
        pos_idxes = np.where(cond)[ 0 ]
        neg_idxes = np.where(np.bitwise_not(cond))[ 0 ]
        labels_dict[ 'age_pos_idxes' ] = pos_idxes
        labels_dict[ 'age_neg_idxes' ] = neg_idxes
        return labels_dict

    def transform_file_list(self, train_file_list, test_file_list, verbose=0):
        train_file_list = np.array(train_file_list)
        test_file_list = np.array(test_file_list)

        train_file_list, train_labels_dict = self.get_data_label(train_file_list)
        # test_label_list, test_file_list = get_data_label(train_file_list)
        print('train_file_list', train_file_list.shape)
        test_file_list, test_labels_dict = self.get_data_label(test_file_list)
        print('test_file_list', test_file_list.shape)

        if verbose > 0:
            def statistic_of_labels(labels, name):
                if isinstance(labels[ 0 ], int):
                    import collections
                    print(collections.Counter(labels))
                else:
                    if not os.path.exists(os.path.join(self.train_dir, 'hist')):
                        os.makedirs(os.path.join(self.train_dir, 'hist'))
                    from matplotlib import pyplot
                    pyplot.figure()
                    pyplot.hist(labels, 20, label=name)
                    pyplot.xlabel('value')
                    pyplot.ylabel(name)
                    pyplot.title(name)
                    pyplot.legend(loc='upper right')
                    path = os.path.join(self.train_dir, 'hist', name + ".png")
                    print(path)
                    pyplot.savefig(path)
                    if verbose > 1:
                        pyplot.show()

            stat_list = [ 'age', 'blur', 'expression', 'race', 'glasses', 'gender', 'beauty', 'angle' ]
            for k in stat_list:
                statistic_of_labels(train_labels_dict[ k ], 'train_' + k)
            for k in stat_list:
                statistic_of_labels(test_labels_dict[ k ], 'test_' + k)

        return train_file_list, train_labels_dict, test_file_list, test_labels_dict

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

        X_train_batch_0, y_train_dict_0, img_index = self.get_data_image_and_label(self.test_bach_size,
                                                                                   self.test_images_list,
                                                                                   self.test_images_list_num,
                                                                                   self.test_images_list_index,
                                                                                   is_training=True)
        self.train_images_list_index = img_index
        return X_train_batch_0, y_train_dict_0

    def get_next_train_batch_data(self):
        if np.random.random() < 0.03:
            self.permutation_train_dataset()

        X_train_batch_0, y_train_dict_0, img_index = self.get_data_image_and_label(self.train_bach_size,
                                                                                   self.train_images_list,
                                                                                   self.train_images_list_num,
                                                                                   self.train_images_list_index,
                                                                                   is_training=True)
        self.train_images_list_index = img_index
        return X_train_batch_0, y_train_dict_0

    def get_attr_label_with_json_file(self, path):
        # result, blur, pitch, yaw, roll, pts4 = None, None, None, None, None
        path_face_attr = path[ :-4 ] + '.detect.json'
        if not os.path.exists(path_face_attr):
            print('get_data_image_and_label no file', path_face_attr)
            exit(0)
        pts68 = self.load_pts_label(path)
        assert pts68 is not None

        with open(path_face_attr) as f:
            face_attr_dict = json.load(f)
            if 'result_num' not in face_attr_dict or face_attr_dict[ 'result_num' ] == 0:
                print('result_num error--', path_face_attr)
                exit(0)

            result = face_attr_dict[ 'result' ][ 0 ]
            blur = result[ 'qualities' ][ 'blur' ]

            pitch = result[ 'pitch' ]
            yaw = result[ 'yaw' ]
            roll = result[ 'roll' ]

            bbox = [ 0 ] * 4
            bbox[ 0 ] = int(result[ 'location' ][ 'left' ])
            bbox[ 1 ] = int(result[ 'location' ][ 'top' ])
            bbox[ 2 ] = int(result[ 'location' ][ 'width' ])
            bbox[ 3 ] = int(result[ 'location' ][ 'height' ])
            pts4 = [ ]
            for point in result[ 'landmark' ]:
                pts4 += [ [ point[ 'y' ], point[ 'x' ] ] ]
        return result, blur, pitch, yaw, roll, pts4, pts68, bbox

    def get_data_image_and_label(self,
                                 sub_batch_size,
                                 file_path_list,
                                 num_data,
                                 img_index=0,
                                 is_training=False):

        img_list, blur_list, pts68_list, angle_list, age_list, \
        beauty_list, expression_list, race_list, pts4_list, \
        pts5_list, glasses_list, gender_list = [ ], [ ], [ ], [ ], [ ], [ ], [ ], [ ], [ ], [ ], [ ], [ ]

        count = 0
        while count < sub_batch_size:
            if img_index == num_data - 1:
                img_index = 0

            path = file_path_list[ img_index ]
            img = cv2.imread(path)
            if img is None or img.shape[ 0 ] <= 0:
                print(path)
                exit(0)

            result, blur, pitch, yaw, roll, pts4, pts68, bbox = self.get_attr_label_with_json_file(path)

            if np.random.random() < 0.8:
                bbox = common.cvt_pts_to_shape(img, pts68)
            if is_training:
                bbox = random_bbox(img, bbox, hw_vari=0.1)
                ext_width_scale = 0.9 + np.random.random() * 1.0
            else:
                ext_width_scale = 1.2
            pts_all = np.concatenate([ pts68, pts4 ], axis=0)

            img, pts_all = common.cut_image_by_bbox(img, bbox, width=self.image_size, ext_width_scale=ext_width_scale,
                                                    pts=pts_all)
            pts68 = pts_all[ :len(pts68) ]
            pts4 = pts_all[ len(pts68): ]
            if img is None or img.shape[ 0 ] <= 0:
                print(path)
                exit(0)

            if is_training:
                img = image_augment_cv(img, aug_proba=0.01, isRgbImage=False, isNormImage=False)

            if self.verbose:
                img = cv2.resize(img, (img.shape[ 1 ] * 4, img.shape[ 0 ] * 4))
                img = common.annotate_bbox(img, bbox * 4)
                img = common.annotate_shapes(img, pts68 * 4)
                img = common.annotate_shapes(img, pts4 * 4)
                s = 'blur ' + str(blur)
                s += ' pitch %.0f yaw %.0f roll %.0f' % (pitch, yaw, roll)
                print(s)
                cv2.putText(img, s, (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7,
                            color=(0, 0, 255))
                cv2.imshow('img', img)
                key = cv2.waitKey(0)
                # if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
                #     exit(0)
                # continue

            img = self.img_attr_preprocess_norm(img)

            img_list += [ img ]
            blur_list += [ blur ]
            pts68_ = pts68.ravel() / image_size
            pts68_list += [ pts68_ ]
            angle_list += [ [ pitch, yaw, roll ] ]
            age_list += [ result[ 'age' ] ]
            beauty_list += [ result[ 'beauty' ] ]
            expression_list += [ int(result[ 'expression' ]) ]
            race_to_int = {'yellow': 0, 'white': 1, 'black': 2, 'arabs': 3}
            race_list += [ race_to_int[ result[ 'race' ] ] ]
            pts4_list += [ pts4.ravel() / image_size ]
            glasses_list += [ int(result[ 'glasses' ]) ]
            pts5_list += [ self.cvt_68pts_to_5pts(np.array(pts68_)) ]
            gender_to_int = {'male': 0, 'female': 1}
            gender_list += [ gender_to_int[ result[ 'gender' ] ] ]

            count += 1
            img_index += 1

        labels_dict = {}
        labels_dict[ 'blur' ] = np.array(blur_list)
        labels_dict[ 'pts68' ] = np.array(pts68_list)
        labels_dict[ 'angle' ] = np.array(angle_list)
        labels_dict[ 'age' ] = np.array(age_list)
        labels_dict[ 'beauty' ] = np.array(beauty_list)
        labels_dict[ 'expression' ] = np.array(expression_list)
        labels_dict[ 'race' ] = np.array(race_list)
        labels_dict[ 'pts4' ] = np.array(pts4_list)
        labels_dict[ 'pts5' ] = np.array(pts5_list)
        labels_dict[ 'glasses' ] = np.array(glasses_list)
        labels_dict[ 'gender' ] = np.array(gender_list)

        # 标签微调
        labels_dict = self.transform_labels_list(labels_dict)
        return np.array(img_list), labels_dict, img_index

    def permutation_train_dataset(self):
        ''' 随机选取难力
        :return:
        '''
        if self.is_select_hard_sample:
            select_index = self.train_bach_size * 10
            file_list_neg_blur = self.train_images_list_bk[
                np.random.permutation(self.train_labels_dict_bk[ 'blur_neg_idxes' ])[ : select_index ] ]
            file_list_pos = self.train_images_list[
                np.random.permutation(self.train_labels_dict_bk[ 'age_pos_idxes' ])[ :select_index ] ]
            file_list_neg = self.train_images_list[
                np.random.permutation(self.train_labels_dict_bk[ 'age_neg_idxes' ])[ :select_index ] ]
            # random
            self.train_images_list = np.concatenate((file_list_pos, file_list_neg, file_list_neg_blur), axis=0)
            self.train_images_list_num = len(self.train_images_list)
            self.train_images_list_index = 0
        else:
            permu_index = np.random.permutation(np.arange(0, self.train_images_list_num))
            self.train_images_list = self.train_images_list[ permu_index ]

    def permutation_test_dataset(self):
        # random
        permu_index = np.random.permutation(np.arange(0, self.test_images_list_num))
        self.test_images_list = self.test_images_list[ permu_index ]


if __name__ == '__main__':
    tmp_folder = r'data_train_attr_temp'
    batch_size = 50
    db = AttrFacefDB(tmp_folder, batch_size)
    db.verbose = True
    for i in range(100):
        db.get_next_test_batch_data()
        db.get_next_train_batch_data()
    pass
