# -*- coding: utf-8 -*-

"""
Created on 2018/11/26

@author: 260157朱发

@decription: 检测视频中眼睛的眨眼次数

@references：Adrian Rosebrock. Eye blink detection with OpenCV, Python, and dlib
"""

import dlib
import cv2
from imutils import face_utils
import imutils
from scipy.spatial import distance as dist

f = 0
headToward = "front"
rightHalfFace = 0
leftHalfFace = 0

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 获取本地视频流
source = "test/videoForTest/headmove.mp4"
# 获取网络视频流
# source = "http://172.28.171.45:8090/"
# 获取直播视频流
# source = "rtmp://172.28.171.74:1935/mylive/room1"
# 从本地默认设备获取视频流
# source = 0
cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:

        frame = imutils.resize(frame, width=1000)
        frame = imutils.rotate(frame, 90)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        rects = detector(gray, 0)
        # 记录帧数
        f += 1

        # 人脸数
        for rect in rects:
            # 提取脸部68个特征点，并转换为numpy数组
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for mark in shape:
                cv2.circle(frame, (mark[0], mark[1]), 2, (0, 255, 0), 1)

            # region for check head toward left or right
            # why we choose 2,14,30,find the answer on 68 face landmark
            keyPoint_right = shape[2]
            keyPoint_left = shape[14]
            keyPoint_nose = shape[30]

            rightHalfFace = dist.euclidean(keyPoint_nose, keyPoint_right)
            leftHalfFace = dist.euclidean(keyPoint_nose, keyPoint_left)

            ratio = min(rightHalfFace, leftHalfFace)/max(rightHalfFace, leftHalfFace)
            # endregion

            # region check head toward up or down(waiting for finish)
            # rightEar = shape[0]
            # leftEar = shape[16]
            # rightEyebrow = shape[17]
            # leftEyebrow = shape[26]
            #
            # rightHalfFace = dist.euclidean(keyPoint_nose, keyPoint_right)
            # leftHalfFace = dist.euclidean(keyPoint_nose, keyPoint_left)
            #
            # ratio = min(rightHalfFace, leftHalfFace)/max(rightHalfFace, leftHalfFace)
            # endregion

        if 0.6 < ratio < 1:
            headToward = "front"
        else:
            headToward = "right" if rightHalfFace > leftHalfFace else "left"

        # 在画面上显示结果以及左右脸的比值
        cv2.putText(frame, "headToward: " + headToward, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "ratio: {:.2f}".format(ratio), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        # 为什么要加“& 0xFF”：https://blog.csdn.net/Addmana/article/details/54604298
        # 按q结束
        if key == ord("q"):
            break
        # 按p暂停循环若干秒
        if key == ord("p"):
            cv2.waitKey(30000)


