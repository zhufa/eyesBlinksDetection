# -*- coding: utf-8 -*-

# 眼睛外接最小矩形

import dlib
import cv2
from imutils import face_utils


# 求外接矩形(水平方向的矩形)
def get_minAreaRect(points):
    # 计算竖直方向眼睛的欧式距离
    min_x = min(points[0][0], points[1][0], points[2][0], points[3][0], points[4][0], points[5][0])
    min_y = min(points[0][1], points[1][1], points[2][1], points[3][1], points[4][1], points[5][1])
    max_x = max(points[0][0], points[1][0], points[2][0], points[3][0], points[4][0], points[5][0])
    max_y = max(points[0][1], points[1][1], points[2][1], points[3][1], points[4][1], points[5][1])
    dist_x = max_x - min_x
    dist_y = max_y - min_y
    return (min_x, min_y), (max_x, max_y), dist_x, dist_y

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# cv2处理图像
img_cv2 = cv2.imread("test1.jpg")

gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 0)

# 人脸数
for rect in rects:
        # 提取脸部68个特征点，并转换为numpy数组
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # 提取左右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        '''
        # 计算眼睛轮廓的凸包并绘出
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img_cv2, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img_cv2, [rightEyeHull], -1, (0, 255, 0), 1)

        minArea = cv2.minAreaRect(leftEye)  # 这里得到的是旋转矩形
        box = cv2.boxPoints(minArea)  # 得到端点
        leftEyeHull = cv2.convexHull(box)
        cv2.drawContours(img_cv2, [leftEyeHull], -1, (0, 255, 0), 1)
        print(box)
        '''
        for point in leftEye:
            pos = (point[00], point[1])
            cv2.circle(img_cv2, pos, 5, color=(0, 255, 0))
        p1, p2, _, _ = get_minAreaRect(leftEye)
        cv2.rectangle(img_cv2, p1, p2, (255, 0, 0), 2)

cv2.namedWindow("img_cv2", 2)
cv2.imshow("img_cv2", img_cv2)
cv2.waitKey(0)