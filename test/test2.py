# -*- coding: utf-8 -*-

# 检测视频中的眼睛

import dlib
import cv2
from imutils import face_utils
import imutils
import matplotlib.pyplot as plt
import numpy as np


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
height, width = 1, 1
f = 0  # 帧数
black_area = 0
ax = []
array_black_area = []
plt.ion()

# 获取本地视频流
source = "videoForTest/577.mp4"
# 获取网络视频流
# source = "http://172.28.171.45:8090/"
# 获取直播视频流
# source = "rtmp://172.28.171.74:1935/mylive/room1"
# 从本地默认设备获取视频流
# source = 0
cap = cv2.VideoCapture(source)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:

        frame = imutils.resize(frame, width=1000)
        # frame = imutils.rotate(frame, 90)
        # 归一化处理
        result = np.zeros(frame.shape, dtype=np.float32)
        cv2.normalize(frame, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        result = np.uint8(result*255.0)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        rects = detector(gray, 0)
        # 记录帧数
        f += 1

        # 人脸数
        for rect in rects:
            # 提取脸部68个特征点，并转换为numpy数组
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # 提取左右眼坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            p1, p2, w, h = get_minAreaRect(leftEye)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 1)

            eye_eara = gray[p1[1]-10:p1[1]+h+10, p1[0]:p1[0]+w]  # 像素坐标的x轴与二维数组的x轴相反，所以先是y:y+h再是x:x+w
            cv2.imshow('eye_eara', eye_eara)
            # 二值化
            ret, thresh1 = cv2.threshold(eye_eara, 70, 255, cv2.THRESH_BINARY)
            cv2.imshow('thresh1', thresh1)
            # 中值滤波
            leftEye_medianBlur = cv2.medianBlur(thresh1, 5)

            cv2.imshow('leftEye_medianBlur', leftEye_medianBlur)

            height, width = leftEye_medianBlur.shape
            for i in range(height):
                for j in range(width):
                    if leftEye_medianBlur[i, j] == 0:
                        black_area += 1

        array_black_area.append(float(black_area)/float(height*width))
        ax.append(f)
        plt.clf()  # 清除之前画的图
        plt.plot(ax, array_black_area)  # x的长度即为y轴数据的个数
        plt.pause(0.1)  # 暂停0.1秒
        black_area = 0
        cv2.imshow("Frame", frame)
        # 根据视频总帧数以及视频总时长计算出每一帧的时间
        # 假如计算出时间为0.04(秒)，则waitKey()填40(ms)
        key = cv2.waitKey(40) & 0xFF
        # 为什么要加“& 0xFF”：https://blog.csdn.net/Addmana/article/details/54604298
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # do a bit of cleanup
plt.ioff()  # 关闭画图的窗口
cv2.destroyAllWindows()
plt.close('all')
plt.plot(range(1, len(array_black_area)+1), array_black_area, color='green', label='original data')
plt.show()