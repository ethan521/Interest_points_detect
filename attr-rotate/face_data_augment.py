import cv2
from time import time
import numpy as np
import os
from common import annotate_points




# 细节增强
from common import scan_image_tree, annotate_points, cut_image_by_bbox


def detail_enhance(current_img, value=None):
    if value is None:
        value = np.random.random()*5 + 1
    value = value * 0.05
    return cv2.detailEnhance(current_img, sigma_s=50, sigma_r=value)

# 边缘保持
def edge_preserve(current_img, value=None):
    if value is None:
        value = np.random.random()*4 + 1
    value = value * 0.05
    return cv2.edgePreservingFilter(current_img, flags=1, sigma_s=50, sigma_r=value)

# 饱和度
def change_saturation(current_img, value=None):
    if value is None:
        value = -30 + np.random.random()*60
    img_hsv = cv2.cvtColor(current_img, cv2.COLOR_BGR2HLS)
    if value > 2:
        img_hsv[:, :, 2] = np.log(img_hsv[:, :, 2] /255* (value - 1)+1) / np.log(value + 1) * 255
    if value < 0:
        img_hsv[:, :, 2] = np.uint8(img_hsv[:, :, 2] / np.log(- value + np.e))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)

# 明度调节
def change_darker(current_img, value=None):
    if value is None:
        value = -13 + np.random.random()*40
    img_hsv = cv2.cvtColor(current_img, cv2.COLOR_BGR2HLS)
    if value > 3:
        img_hsv[:, :, 1] = np.log(img_hsv[:, :, 1] /255* (value - 1)+1) / np.log(value + 1) * 255
    if value < 0:
        img_hsv[:, :, 1] = np.uint8(img_hsv[:, :, 1] / np.log(- value + np.e))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)


def crop_image(img, x0, y0, w, h):
    """
    定义裁剪函数
    :param img: 要处理的图片
    :param x0: 左上角横坐标
    :param y0: 左上角纵坐标
    :param w: 裁剪宽度
    :param h: 裁剪高度
    :return: 裁剪后的图片
    """
    return img[x0:x0+w, y0:y0+h]

def random_pad(img, area_ratio, hw_vari):
    """
    定义随机pad函数
    :param img: 要处理的图片
    :param area_ratio: pad画面占原图片的比例
    :param hw_vari: 扰动占原宽高的比例
    :return: 裁剪后的图片
    """
    w, h = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
    color = int(np.random.random()*200)
    w_pad = int(np.round(w * np.sqrt(area_ratio * hw_mult)))
    h_pad = int(np.round(h * np.sqrt(area_ratio / hw_mult)))
    top, bottom, left, right = h_pad, h_pad, w_pad, w_pad
    pad_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return pad_image

def random_crop(img, area_ratio, hw_vari):
    """
    定义随机裁剪函数
    :param img: 要处理的图片
    :param area_ratio: 裁剪画面占原图片的比例
    :param hw_vari: 扰动占原宽高的比例
    :return: 裁剪后的图片
    """
    w, h = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta

    # 裁剪宽度
    w_crop = int(np.round(w * np.sqrt(area_ratio * hw_mult)))
    # 裁剪宽度不能大于原图宽度
    if w_crop > w:
        w_crop = w
    # 裁剪高度
    h_crop = int(np.round(h * np.sqrt(area_ratio / hw_mult)))
    if h_crop > h:
        h_crop = h

    # 随机生成左上角的位置
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)

    return crop_image(img, x0, y0, w_crop, h_crop)


def rotate_image(img, angle, crop):
    """
    定义旋转函数
    :param img: 要处理的图片
    :param angle: 图片旋转的角度
    :param crop: 是否对旋转后出现的黑边进行裁剪
    :return: 旋转后的图片
    """
    w, h = img.shape[:2]

    # 旋转角度的周期是360°
    angle %= 360

    # 计算仿射变换矩阵
    M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

    # 如果需要裁剪黑边
    if crop:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop

        # 角度转化为弧度
        theta = angle_crop * np.pi / 180

        # 计算高宽比
        hw_ratio = float(h) / float(w)

        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta

        # 计算分母项中相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 分母项
        denominator = r * tan_theta + 1

        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry


def rotate_points(points, centerX, centerY, angle):
    points = np.array([rotate_point(l[0], l[1], centerX, centerY, angle) for l in points])
    return points


def random_rotate(img, angle_vari, p_crop):
    """
    定义随机旋转函数
    :param img: 要处理的图片
    :param angle_vari: 旋转角度的范围
    :param p_crop: 要进行裁剪黑边的图片所占的比例
    :return: 随机旋转的图片
    """
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True

    return rotate_image(img, angle, crop)


def hsv_transform(img, hue_delta, sat_mult, val_mult):
    """
    定义hsv转换函数
    :param img: 要处理的图片
    :param hue_delta: 色调变化
    :param sat_mult: 饱和度变化
    :param val_mult: 明暗度变化
    :return: hsv变化后的图片
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255

    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    """
    随机hsv变换
    :param img: 要处理的图片
    :param hue_vari: 色调比例变化范围
    :param sat_vari: 饱和度比例变化范围
    :param val_vari: 明暗度比例变化范围
    :return: 随机hsv变换后的图片
    """
    hue_delta = np.random.uniform(-hue_vari, hue_vari)
    sat_mult = np.random.uniform(-sat_vari, sat_vari) + 1
    val_mult = np.random.uniform(-val_vari, val_vari) + 1

    return hsv_transform(img, hue_delta, sat_mult, val_mult)


def gamma_transform(img, gamma):
    """
    定义gamma变换函数
    :param img: 要处理的函数
    :param gamma: gamma系数
    :return: gamma变换后的图片
    """
    gamma_table = [np.power(x / 255, gamma) * 255 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)

def fill_rect(img, pt1, pt2, color, transparency=0.2):
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    pts = np.array([[[x1,y1], [x1,y2], [x2,y2], [x2,y1]]])
    img_fill = img.copy()
    cv2.fillPoly(img_fill, pts, color)
    img = cv2.addWeighted(img, transparency, img_fill, 1-transparency, 1.0)
    return img

def draw_liuhai(img):
    # img = cv2.imread('./im_160x160.jpg')
    img = np.copy(img)
    # img = cv2.resize(img, (100, 100))
    coord_point = np.array([[ 34.19, 46.16], [ 65.65, 45.98], [ 50.12, 82.40]], dtype=np.float32)
    coord_point = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32)
    width = img.shape[0]
    coord_point = coord_point / 100 * width
    h_diff = int(np.random.random()*0.03*width)
    left_eye = (int(coord_point[0][0]), int(coord_point[0][1] + h_diff))
    right_eye = (int(coord_point[1][0]), int(coord_point[1][1] + h_diff))
    glass_w = (0.15 + np.random.random()*0.05) * width
    glass_w = int(glass_w)
    glass_h = (0.1 + np.random.random()*0.10) * width
    glass_h = int(glass_h)
    color = int(np.random.random()*100)
    hh = 0.1 * width
    pt1 = (left_eye[0]-glass_w, int(max(left_eye[1]-glass_h-hh,0)))
    pt2 = (right_eye[0]+glass_w, int(right_eye[1]-hh))
    img = fill_rect(img, pt1, pt2, color, transparency=0.1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('img_glass.jpg', img)
    return img

def draw_beard(img):
    # img = cv2.imread('./im_160x160.jpg')
    img = np.copy(img)
    # img = cv2.resize(img, (100, 100))
    coord_point = np.array([[ 34.19, 46.16], [ 65.65, 45.98], [ 50.12, 82.40]], dtype=np.float32)
    coord_point = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32)
    width = img.shape[0]
    coord_point = coord_point / 100 * width
    h_diff = int(np.random.random()*0.03*width)
    left_eye = (int(coord_point[0][0]), int(coord_point[0][1] + h_diff))
    right_eye = (int(coord_point[1][0]), int(coord_point[1][1] + h_diff))
    glass_w = (0.02 + np.random.random()*0.04) * width
    glass_w = int(glass_w)
    glass_h = (0.1 + np.random.random()*0.10) * width
    glass_h = int(glass_h)
    color = int(np.random.random()*100)
    hh = 0.43 * width
    pt1 = (left_eye[0]-glass_w, int(max(left_eye[1]-glass_h+hh,0)))
    pt2 = (right_eye[0]+glass_w, int(min(right_eye[1]+hh,width)))
    img = fill_rect(img, pt1, pt2, color, transparency=0.5)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('img_glass.jpg', img)
    return img

def draw_glass(img):
    # img = cv2.imread('./im_160x160.jpg')
    img = np.copy(img)
    # img = cv2.resize(img, (100, 100))
    coord_point = np.array([[ 34.19, 46.16], [ 65.65, 45.98], [ 50.12, 82.40]], dtype=np.float32)
    coord_point = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32)

    width = img.shape[0]
    coord_point = coord_point / 100 * width
    # img = annotate_shapes(img, coord_point, verbose=1)
    h_diff = int(np.random.random()*0.03*width)
    left_eye = (int(coord_point[0][0]), int(coord_point[0][1] + h_diff))
    right_eye = (int(coord_point[1][0]), int(coord_point[1][1] + h_diff))
    # cv2.circle(img, left_eye, 10, color=(0, 0, 0), thickness=4)
    # cv2.circle(img, right_eye, 10, color=(0, 0, 0), thickness=4)
    glass_h = (0.05 + np.random.random()*0.10) * width
    glass_w = (0.15 + np.random.random()*0.05) * width
    glass_h = int(glass_h)
    glass_w = int(glass_w)
    thickness = 1 + np.random.random()*5
    thickness = int(thickness)
    # print(thickness)
    if np.random.random() < 0.2:
        cv2.ellipse(img, left_eye, (glass_w, glass_h),0,0,360,0, thickness=thickness)
        cv2.ellipse(img, right_eye, (glass_w, glass_h),0,0,360,0, thickness=thickness)
    else:
        color = int(np.random.random()*255)
        glass_w -= 3
        glass_h += 1
        pt1 = (left_eye[0]-glass_w, left_eye[1]-glass_h)
        pt2 = (left_eye[0]+glass_w, left_eye[1]+glass_h)
        cv2.rectangle(img, pt1, pt2, color=color, thickness=thickness)
        img = fill_rect(img, pt1, pt2, color, transparency=0.6)
        pt1 = (right_eye[0]-glass_w, right_eye[1]-glass_h)
        pt2 = (right_eye[0]+glass_w, right_eye[1]+glass_h)
        cv2.rectangle(img, pt1, pt2, color=color, thickness=thickness)
        img = fill_rect(img, pt1, pt2, color, transparency=0.6)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('img_glass.jpg', img)
    return img

def cvt_68pts_to_5pts(points68):
    points5 = np.array([
        np.mean(points68[36:42], axis=0),  # left eye
        np.mean(points68[42:48], axis=0),  # right eye
        np.mean(points68[30:35], axis=0),  # node
        points68[48],  # left mouth
        points68[54]]  # right mouth
    )
    return points5

def cvt_5pts_to_3pts(point):
    point = point[:, ::-1]
    mouth = point[3:, :].mean(axis=0)
    coord_point = np.array([point[0], point[1], mouth], dtype=np.float32)
    return coord_point

def cvt_68pts_to_3pts(points68):
    points = cvt_68pts_to_5pts(points68)
    points = cvt_5pts_to_3pts(points)
    return points

random_occlusion_init = False
random_occlusion_path_list = []

def get_random_image_for_occlusion():
    global random_occlusion_init
    global random_occlusion_path_list
    if not random_occlusion_init:
        random_occlusion_init = True
        image_dir = r'F:\data\face-attr\zhangbiao\48\negative'
        image_dir = r'F:\data\face-attr\zhangbiao\96\negative_48_hard'
        image_dir = r'F:\data\face-attr\zhangbiao\96\positive_remove_pitch'
        image_dir = r'F:\data\face-attr\louyu-capture\downloadimg-20180911'
        image_dir = r'D:\data\face-attr\test\test_occlusion'
        image_dir = r'F:\data\face-attr\zhangbiao\96\part'
        random_occlusion_path_list = scan_image_tree(image_dir)
        np.random.shuffle(random_occlusion_path_list)
    idx = np.random.randint(len(random_occlusion_path_list))
    img = cv2.imread(random_occlusion_path_list[idx])
    img = random_crop(img, 0.2, 0.3)
    # print(img.shape)
    return img

JAW_POINTS        = list(range(0, 17))
FACE_POINTS       = list(range(17, 68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS  = list(range(22, 27))
NOSE_POINTS       = list(range(27, 35))
RIGHT_EYE_POINTS  = list(range(36, 42))
LEFT_EYE_POINTS   = list(range(42, 48))
MOUTH_POINTS      = list(range(48, 61))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
    JAW_POINTS,
    ]
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
    ]

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks, rand=None):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    draw_convex_hull(im, landmarks, color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    FEATHER_AMOUNT = 11
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im

def correct_colours(im1, im2, landmarks1):
    # Amount of blur to use during colour correction, as a fraction of the
    # pupillary distance.
    COLOUR_CORRECT_BLUR_FRAC = 0.6
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))

def gen_swap_face(im1, landmarks1, im2):
    landmarks1 = np.copy(landmarks1)
    landmarks1 = landmarks1[:, ::-1]
    landmarks1 = np.matrix(landmarks1.astype(int))
    rand = np.random.random()
    mask = get_face_mask(im1, landmarks1, rand)
    output_im = im1 * (1.0 - mask) + im2 * mask
    output_im = output_im.astype(np.uint8)
    if False:
        im3 = annotate_points(im1, landmarks1)
        cv2.imshow('mask1', mask)
        cv2.imshow('im1', im1)
        cv2.imshow('im2', im2)
        cv2.imshow('im3', im3)
        cv2.imshow('output_im', output_im)
        key = cv2.waitKey(0)
        if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
            exit(0)
    return output_im

def random_occlusion_keypoints(img, points=None, occ_proba=0.2):
    # img = cv2.imread('./im_160x160.jpg')
    img = np.copy(img)
    # img = cv2.resize(img, (100, 100))
    if points is not None:
        point = points
        if len(points) == 68:
            point = cvt_68pts_to_3pts(point)
        elif len(points) == 5:
            point = cvt_5pts_to_3pts(point)
        coord_point = point
    else:
        coord_point = np.array([[ 34.19, 46.16], [ 65.65, 45.98], [ 50.12, 82.40]], dtype=np.float32)
        coord_point = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32)
        width = img.shape[0]
        coord_point = coord_point / 100 * width
        points = coord_point

    eye_width = coord_point[1][0] - coord_point[0][0]
    # img = annotate_shapes(img, coord_point, verbose=1)
    left_eye = (int(coord_point[0][0]), int(coord_point[0][1]))
    right_eye = (int(coord_point[1][0]), int(coord_point[1][1]))
    mouth = (int(coord_point[2][0]), int(coord_point[2][1]))
    # cv2.circle(img, left_eye, 10, color=(0, 0, 0), thickness=4)
    # cv2.circle(img, right_eye, 10, color=(0, 0, 0), thickness=4)

    img_h, img_w = img.shape[0], img.shape[1]
    occ_labels = [0, 0, 0]
    minr = 0.05
    maxr = 0.95
    if left_eye[0] < minr * img_w or left_eye[1] < minr * img_w:
        occ_labels[0] = 1
    if right_eye[0] > maxr * img_w or right_eye[1] > maxr * img_w:
        occ_labels[1] = 1
    if mouth[1] > maxr * img_h:
        occ_labels[2] = 1
    # print('minr, maxr', minr * img_w, maxr * img_w, maxr * img_h)
    # print('coord_point', img.shape, coord_point)

    def fill_image_random(img, pt1, pt2):
        img = np.copy(img)
        img_occ = get_random_image_for_occlusion()
        pt1[0] = max(pt1[0], 0)
        pt1[1] = max(pt1[1], 0)
        pt2[0] = min(pt2[0], img.shape[1])
        pt2[1] = min(pt2[1], img.shape[0])
        glass_h = pt2[1] - pt1[1]
        glass_w = pt2[0] - pt1[0]
        if glass_w <= 0 or glass_h <= 0:
            # print(coord_point)
            # print(pt1, pt2)
            # cv2.imshow('img', img)
            # key = cv2.waitKey(0)
            return img
        # img_re = cv2.resize(img, (img_occ.shape[1], img_occ.shape[0]))
        # img_occ = correct_colours(img_re, img_occ, points)
        img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = cv2.resize(img_occ, (glass_w, glass_h))
        return img

    # occ left eye
    if np.random.random() < occ_proba and occ_labels[0] == 0:
        occ_labels[0] = 1
        glass_h = int((0.25 + np.random.random()*0.25) * eye_width)
        glass_w = int((0.30 + np.random.random()*0.25) * eye_width)
        pt1 = [left_eye[0]-glass_w, left_eye[1]-glass_h]
        pt2 = [left_eye[0]+glass_w, left_eye[1]+glass_h]
        img_ = fill_image_random(img, pt1, pt2)
        if np.random.random() < 0.1 and len(points) == 68:
            img = gen_swap_face(img, points[RIGHT_EYE_POINTS], img_)
        else:
            img = img_

    # occ right eye
    if np.random.random() < occ_proba and occ_labels[0] == 0:
        occ_labels[1] = 1
        glass_h = int((0.25 + np.random.random()*0.25) * eye_width)
        glass_w = int((0.30 + np.random.random()*0.25) * eye_width)
        pt1 = [right_eye[0]-glass_w, right_eye[1]-glass_h]
        pt2 = [right_eye[0]+glass_w, right_eye[1]+glass_h]
        img_ = fill_image_random(img, pt1, pt2)
        if np.random.random() < 0.1 and len(points) == 68:
            img = gen_swap_face(img, points[LEFT_EYE_POINTS], img_)
        else:
            img = img_

    # occ mouth
    if np.random.random() < occ_proba and occ_labels[0] == 0:
        occ_labels[2] = 1
        glass_h = int((0.25 + np.random.random()*0.30) * eye_width)
        glass_w = int((0.35 + np.random.random()*0.30) * eye_width)
        pt1 = [mouth[0]-glass_w, mouth[1]-glass_h]
        pt2 = [mouth[0]+glass_w, mouth[1]+glass_h]
        img_ = fill_image_random(img, pt1, pt2)
        if np.random.random() < 0.1 and len(points) == 68:
            img = gen_swap_face(img, points[MOUTH_POINTS], img_)
        else:
            img = img_

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('img_glass.jpg', img)
    return img, occ_labels

def random_occlusion(img, area_ratio=0.05, hw_vari=0.5):
    w, h = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
    # 裁剪宽度
    w_crop = int(np.round(w * np.sqrt(area_ratio * hw_mult)))
    # 裁剪宽度不能大于原图宽度
    if w_crop > w:
        w_crop = w
    # 裁剪高度
    h_crop = int(np.round(h * np.sqrt(area_ratio / hw_mult)))
    if h_crop > h:
        h_crop = h
    # 随机生成左上角的位置
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)
    pt1 = (x0, y0)
    pt2 = (x0+w_crop, y0+h_crop)
    color = int(np.random.random()*100)
    img = fill_rect(img, pt1, pt2, color, transparency=0.1)
    return img

def random_bbox(img, bbox, hw_vari=0.3):
    for _ in range(10):
        w, h = bbox[2], bbox[3]
        x1 = bbox[0] + w * (np.random.uniform(-hw_vari, hw_vari))
        y1 = bbox[1] + h * (np.random.uniform(-hw_vari, hw_vari))
        x2 = bbox[0] + bbox[2] + w * (np.random.uniform(-hw_vari, hw_vari))
        y2 = bbox[1] + bbox[3] + h * (np.random.uniform(-hw_vari, hw_vari))
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img.shape[1])
        y2 = min(y2, img.shape[0])
        new_w = x2-x1
        new_h = y2-y1
        if x1 < img.shape[1] * 0.8 and y1 < img.shape[0] * 0.8 and new_w > w*0.3 and new_h > h*0.3:
            bbox = [x1, y1, new_w, new_h]
            break
    return np.array(bbox)

def image_augment_cv(img, aug_proba=1.0, isRgbImage=False, isNormImage=False):
    img = img.copy()
    img_size = img.shape[:2]
    if isNormImage:
        img *= 128.
        img += 128.
        img = img.astype(np.uint8)
    grayscale = img.shape[2] == 1
    if grayscale:
        img = np.concatenate([img, img, img], axis=2)
    if isRgbImage:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    info = ''
    if np.random.random() < aug_proba*0.01:
        info += 'detail_enhance '
        img = detail_enhance(img, np.random.random()*4)
    if np.random.random() < aug_proba*0.001:
        info += 'draw_glass '
        img = draw_glass(img)
    if np.random.random() < aug_proba*0.0005:
        info += 'draw_liuhai '
        img = draw_liuhai(img)
    if np.random.random() < aug_proba*0.0005:
        info += 'draw_beard '
        img = draw_beard(img)
    if np.random.random() < aug_proba*0.003:
        info += 'edge_preserve '
        img = edge_preserve(img, np.random.random()*3)
    if np.random.random() < aug_proba*0.1:
        info += 'random_occlusion '
        img = random_occlusion(img)
    if np.random.random() < aug_proba*0.1:
        info += 'change_saturation '
        img = change_saturation(img, -20 + np.random.random()*40)
    if np.random.random() < aug_proba:
        info += 'change_darker '
        img = change_darker(img, -8 + np.random.random()*16)
    if np.random.random() < aug_proba and False:
        info += 'random_rotate '
        img = random_rotate(img, 4, True)
    if np.random.random() < aug_proba and False:
        info += 'random_pad '
        img = random_pad(img, 0.005, 0.15)
    if np.random.random() < aug_proba and False:
        info += 'random_crop '
        img = random_crop(img, 0.7, 0.5)
    if np.random.random() < aug_proba*0.3:
        info += 'random_grayscale '
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if np.random.random() < aug_proba*0.3:
        info += 'random_invert '
        img = 255 - img
    if np.random.random() < aug_proba*0.1:
        info += 'random_bgr '
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if isRgbImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=3)
    if len(info) > 0:
        # print(info)
        pass
    if isNormImage:
        img = img.astype(float)
        img = img - 128.
        img = img / 128.
    return img


def demo_rotate():
    img_path = r'D:\chenkai\facenet-dev-all\src\Aaron_Eckhart_0001.jpg'
    img_path = r'E:\data\face-1vs1\capture_glass\im_160.jpg'
    img_raw = cv2.imread(img_path)

    while True:
        t1 = time()
        angle = 45 - np.random.random() * 90
        points_raw = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32) / 100 * img_raw.shape[0]
        bbox = np.array([10, 10, img_raw.shape[1]-10, img_raw.shape[0]-10])

        img = np.copy(img_raw)
        points = np.copy(points_raw)

        bbox = random_bbox(img, bbox, hw_vari=0.5)
        ext_width_scale = 1.2
        img, points = cut_image_by_bbox(img, bbox, width=160, ext_width_scale=ext_width_scale,
                                                pts=points, swithxy=True)
        img, occ_labels = random_occlusion_keypoints(img, points, occ_proba=0.3)
        print('occ_labels', occ_labels)
        # img = rotate_image(img, angle, False)
        # points = rotate_points(points, img.shape[1]//2, img.shape[0]//2, angle)
        img_raw_disp = annotate_points(img_raw, points_raw)
        cv2.imshow('img', img_raw_disp)
        img = annotate_points(img, points)
        cv2.imshow('img_aug.jpg', img)
        key = cv2.waitKey(0)
        if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
            break



def demo_aug():
    img_path = r'D:\chenkai\facenet-dev-all\src\Aaron_Eckhart_0001.jpg'
    img_path = r'E:\data\face-1vs1\capture_glass\im_160.jpg'
    img_raw = cv2.imread(img_path)

    while True:
        t1 = time()
        for _ in range(100):
            img = image_augment_cv(img_raw, aug_proba=1.)
        print(time()-t1, 's')
        cv2.imshow('img', img)
        cv2.imwrite('img_aug.jpg', img)
        key = cv2.waitKey(0)
        if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
            break


def augment_dataset():
    input_dir = r'E:\data\face-1vs1\test-5w-3d\ISOK-160'
    input_dir = r'E:\data\face-recognition\real\nanningyouji-160'
    input_dir = r'E:\data\face-recognition\MS-Celeb-1M\MsCelebV1-Faces-Aligned-160-Clean-Relabel-128Vec'
    input_dir = r'E:\data\face-1vs1\PhotoCategories-mxnet\train'
    import facenet
    dataset = facenet.get_dataset(input_dir)
    print(input_dir)

    nrof_classes = len(dataset)

    t1 = time()
    if False:
        import random
        random.shuffle(dataset)
    for ci, cls in enumerate(dataset):
        for image_path in cls.image_paths[:5]:
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            if 'glass' in filename:
                continue
            try:
                img = cv2.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if not isinstance(img, np.ndarray):
                    continue
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:,:,0:3]
                # vs = filename.split('_')
                # yaw = int(vs[3])
                yaw = 0
                if yaw < 40:
                    output_class_dir = os.path.split(image_path)[0]
                    output_filename = os.path.join(output_class_dir, 'glass_'+filename+'.jpg')
                    # print(image_path)
                    if os.path.exists(output_filename):
                        continue
                    img = draw_glass(img)
                    cv2.imwrite(output_filename, img)

        print('%.0f s - %.0f s  %d / %d %s'% (
            time()-t1, (time()-t1)/(ci+1)*(nrof_classes-ci), ci, nrof_classes, cls.name))

if __name__ == '__main__':
    # augment_dataset()
    # demo_aug()
    demo_rotate()
