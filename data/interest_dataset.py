# coding=utf-8

import os
import numpy as np
import cv2
import pickle
import sys
import copy
from face_data_augment import *
# from utils_file import *
from common import scan_image_tree_get_relative_path , scan_folder_tree_get_relative_path , scan_image_tree
from common import annotate_bbox , annotate_points

image_size = 100

from data.database import DataBaseDB
import common
import json
import cv2
import time


# from  face_data_augment import random_bbox, image_augment_cv


class InterestDB ( DataBaseDB ):
	def __init__(self , data_batch_size=512):
		'''
		:param tmp_folder:
		:param data_batch_size:

		init AttrFacefDB done
		test num:20119
		train num:216253
		'''
		super ( InterestDB , self ).__init__ ()
		self.data_name = "InterestDB"
		self.image_size = 36
		self.image_size_half = 36
		self.root_fl = '/Users/a1/ws_dl/mgtvai/data_pointstrack/PointsTracking'
		self.train_dir = r"..\running_tmp\interest_model\20200228_interests_model"
		self.dataset_folder = self.root_fl + r'D:\data\face-attr\louyu-capture'
		
		self.val_txt = [ self.root_fl + r'/val.txt' ]
		self.test_txt = [ self.root_fl + r'/test_a.txt' ]
		self.train_txt = [ self.root_fl + r'/research_datasets_1.txt' ,
		                   self.root_fl + r'/research_datasets_2.txt' ]
		
		train_images_list , train_labels_list = self.load_data_file_list ( self.train_txt )
		self.train_images_list = np.array ( train_images_list )
		self.train_labels_list = np.array ( train_labels_list )
		self.train_images_list_num = len ( self.train_images_list )
		self.train_images_list_index = 0
		
		# self.test_images_list,self.test_labels_dict = self.load_data_file_list(self.test_txt)
		test_images_list , test_labels_list = self.load_data_file_list ( self.val_txt )
		self.test_images_list = np.array ( test_images_list )
		self.test_labels_list = np.array ( test_labels_list )
		self.test_images_list_num = len ( self.test_images_list )
		self.test_images_list_index = 0
		
		self.test_bach_size = data_batch_size
		self.train_bach_size = data_batch_size
		
		self.verbose = False
		self.aug_proba = 1.0
		
		self.is_select_hard_sample = False
		if self.is_select_hard_sample:
			self.train_images_list_bk = self.train_images_list
			self.train_labels_list_bk = self.train_labels_list
			self.train_images_list_num_bk = self.train_images_list_num
			self.train_images_list_index_bk = self.train_images_list_index
		
		self.permutation_test_dataset ()
		self.permutation_train_dataset ()
		
		print ( '-' * 50 )
		print ( 'init {} done'.format ( self.data_name ) )
		print ( 'test num:{}'.format ( self.test_images_list_num ) )
		print ( 'train num:{}'.format ( self.train_images_list_num ) )
		print ( '-' * 50 )
		pass
	
	def load_data_file_list(self , txt_list , is_reload=True):
		image_list , coords_list = [ ] , [ ]
		for i , label_path in enumerate ( txt_list ):
			print ( '>> load txt: {}'.format ( label_path ) )
			img_list , pts_list = readfile ( label_path )
			image_list.extend ( img_list )
			coords_list.extend ( pts_list )
		return image_list , coords_list
		pass
	
	def get_next_test_batch_data(self):
		if np.random.random () < 0.05:
			self.permutation_test_dataset ()
		
		X_train_batch_0 , \
		y_train_dict_0 , \
		img_index = self.get_data_image_and_label (
			self.test_bach_size ,
			self.test_images_list ,
			self.test_labels_list ,
			self.test_images_list_num ,
			self.test_images_list_index ,
			is_training=True )
		
		self.train_images_list_index = img_index
		return X_train_batch_0 , y_train_dict_0
	
	def get_next_train_batch_data(self):
		if np.random.random () < 0.03:
			self.permutation_train_dataset ()
		
		X_train_batch_0 , \
		y_train_batch_0 , \
		img_index = self.get_data_image_and_label (
			self.train_bach_size ,
			self.train_images_list ,
			self.train_labels_list ,
			self.train_images_list_num ,
			self.train_images_list_index ,
			is_training=True )
		self.train_images_list_index = img_index
		return X_train_batch_0 , y_train_batch_0
	
	def get_input_img_and_label(self , img_raw , pts4_raw):
		'''
		:param img_raw: 原图数据
		:param pts4: 4个点坐标[x1,y1,x2,y2,x3,y3,x4,y4]
		:return:
		'''
		
		if True:
			pt_1 = (pts4_raw[ 0 ] , pts4_raw[ 1 ])
			pt_2 = (pts4_raw[ 2 ] , pts4_raw[ 3 ])
			pt_3 = (pts4_raw[ 4 ] , pts4_raw[ 5 ])
			pt_4 = (pts4_raw[ 6 ] , pts4_raw[ 7 ])
			pts = np.array ( [ [ pt_1 , pt_2 , pt_3 , pt_4 ] ] , dtype=np.int32 )
			cv2.fillPoly ( img_raw , pts , 255 )
			cv2.imshow ( 'img_raw' , copy.copy ( img_raw ) )
			
		tmp_imgs , tmp_pts_list = [ ] , [ ]
		pts4 = np.array ( pts4_raw , dtype=int ).reshape ( (4 , 2) )
		for i in range ( 4 ):
			# 待加入数据增广, 随机pad
			x = pts4[ i ][ 0 ]
			y = pts4[ i ][ 1 ]
			start_x = min ( x , max ( 0 , x - self.image_size_half ) )
			start_y = min ( y , max ( 0 , y - self.image_size_half ) )
			print ( 'shape:{}, x:{}, start_x:{},y:{}, start_y:{}'.format (img_raw.shape, x , start_x , y , start_y ) )
			img = img_raw[
			        start_y:start_y + self.image_size,
					start_x:start_x + self.image_size
			      ]
			
			cv2.imshow ( '{}'.format(i) , copy.copy(img) )
			cv2.waitKey ( 0 )
			pts = [ pts4[ i ][ 0 ] - start_x , pts4[ i ][ 1 ] - start_y ]
			tmp_imgs.append ( img )
			tmp_pts_list.append ( pts )
		return tmp_imgs , tmp_pts_list
		
		pass
	
	def get_data_image_and_label(self ,
	                             sub_batch_size ,
	                             file_path_list ,
	                             label_list ,
	                             num_data ,
	                             img_index=0 ,
	                             is_training=False):
		
		img_list , pts_list = [ ] , [ ]
		count = 0
		while count < sub_batch_size + 10:
			if img_index == num_data - 1:
				img_index = 0
			
			path = file_path_list[ img_index ]
			img = cv2.imread ( path )
			if img is None or img.shape[ 0 ] <= 0:
				print ( path )
				exit ( 0 )
			
			pts_4 = label_list[ img_index ]
			# 获取单张图生成的小图，以及标签点：预测单个点x,y,范围(0,1)
			tmp_imgs , tmp_pts_list = self.get_input_img_and_label ( img , pts_4 )
			
			count += len ( tmp_imgs )
			img_index += 1
		
		return np.array ( img_list[ :sub_batch_size ] ) , np.array ( pts_list[ :sub_batch_size ] ) , img_index
	
	def permutation_train_dataset(self):
		permu_index = np.random.permutation ( np.arange ( 0 , self.train_images_list_num ) )
		self.train_images_list = self.train_images_list[ permu_index ]
		self.train_labels_list = self.train_labels_list[ permu_index ]
	
	def permutation_test_dataset(self):
		# random
		permu_index = np.random.permutation ( np.arange ( 0 , self.test_images_list_num ) )
		self.test_images_list = self.test_images_list[ permu_index ]
		self.test_labels_list = self.test_labels_list[ permu_index ]


LINE_ITEMS_LEN = 9
COORDS_LEN = 8
LINE_ERROR_STR = "行的格式不正确，正确格式如: " \
                 "\n\"n x1,y1 x2,y2, x3,y3 x4,y4\"\n" \
                 "[备注：所有项都应该是数字格式]\n"
NUM_FRAMES_ERROR_STR = "提供的结果文件帧数目与视频帧数不一致，请检查"
FRAME_IDX_ERROR_STR = "行号需与帧序号一致"


def readfile(filename):
	image_list , coords_list = [ ] , [ ]
	with open ( filename , 'r' ) as fr:
		lines = fr.readlines ()
		for idx , line in enumerate ( lines ):
			frame_idx , coords = readline ( line )
			frame_idx = frame_idx.replace ( "\\" , "/" )
			frame_idx = frame_idx.replace ( 'D:/data/mgtvai_game/PointsTracking' , root_fl )
			assert os.path.exists ( frame_idx )
			coords_list.append ( coords )
			image_list.append ( frame_idx )
	return np.array ( image_list ) , np.array ( coords_list )


def readline(line):
	items = line.strip ( '\n' ).split ()
	# print(len(items))
	if len ( items ) != LINE_ITEMS_LEN:
		raise Exception ( LINE_ERROR_STR )
	frame_idx = items[ 0 ]
	try:
		# points = [tuple([float(i) for i in xy_str.split(',')]) for xy_str in items[1:]]
		points = [ float ( xy_str ) for xy_str in items[ 1: ] ]
		points = np.array ( points ).reshape ( (-1 ,) )
		if len ( points ) != COORDS_LEN:
			raise Exception ( "需要确保每行只有四点" )
	
	except Exception as e:
		raise Exception ( LINE_ERROR_STR )
	return frame_idx , points


root_fl = '/Users/a1/ws_dl/mgtvai/data_pointstrack/PointsTracking'
if __name__ == '__main__':
	# tmp_folder = r'data_train_attr_temp'
	batch_size = 50
	db = InterestDB ( batch_size )
	db.verbose = True
	for i in range ( 100 ):
		db.get_next_test_batch_data ()
		db.get_next_train_batch_data ()
	
	# label_path = r'D:\data\mgtvai_game\PointsTracking\val.txt'
	# label_path = root_fl + r'/val.txt'
	# label_path = root_fl + r'/test_a.txt'
	# label_path = root_fl + r'/research_datasets_1.txt'
	# label_path = root_fl + r'/research_datasets_2.txt'
	# image_list , coords_list = readfile ( label_path )
	#
	# for i , img_path in enumerate ( image_list ):
	# 	img_path = img_path.replace ( "\\" , "/" )
	# 	img_path = img_path.replace ( 'D:/data/mgtvai_game/PointsTracking' , root_fl )
	# 	img = cv2.imread ( img_path )
	# 	if img is None:
	# 		continue
	# 	print ( img_path )
	# 	coords = coords_list[ i ]
	# 	pt_1 = (coords[ 0 ] , coords[ 1 ])
	# 	pt_2 = (coords[ 2 ] , coords[ 3 ])
	# 	pt_3 = (coords[ 4 ] , coords[ 5 ])
	# 	pt_4 = (coords[ 6 ] , coords[ 7 ])
	# 	pts = np.array ( [ [ pt_1 , pt_2 , pt_3 , pt_4 ] ] , dtype=np.int32 )
	# 	cv2.fillPoly ( img , pts , 255 )
	# 	cv2.resize ( img , (640 , 360) , img )  # 1080p --> 720p
	# 	cv2.imshow ( 'img' , img )
	# 	cv2.waitKey ( 1 )
	# 	# break
	# 	time.sleep ( 1 )
	# pass
