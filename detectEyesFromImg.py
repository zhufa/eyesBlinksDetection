# -*- coding: utf-8 -*-

# 检测图片中的眼睛

import dlib
from PIL import Image
import numpy as np
import cv2
from imutils import face_utils

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cv2处理图像
img_cv2 = cv2.imread("test/imgForTest/test4.jpeg")

# PIL处理图像
img_PIL = Image.open("test/imgForTest/test4.jpeg").convert('RGBA')

# 灰度
img_gray = np.array(img_PIL.convert('L'))

rects = detector(img_gray, 0)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


for rect in rects:
    # 提取脸部68个特征点，并转换为numpy数组
    shape = predictor(img_gray, rect)
    shape = face_utils.shape_to_np(shape)
    # 提取左右眼坐标
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    # 计算眼睛轮廓的凸包并绘出
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(img_cv2, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(img_cv2, [rightEyeHull], -1, (0, 255, 0), 1)

cv2.imshow("img_cv2", img_cv2)
cv2.waitKey()