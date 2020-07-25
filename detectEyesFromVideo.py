# -*- coding: utf-8 -*-

# 检测视频中的眼睛

import dlib
import cv2
from imutils.video import FileVideoStream
from imutils import face_utils
import imutils
import time

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

vs = FileVideoStream("test/videoForTest/57.mp4").start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)  # 此处为什么要sleep

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 灰度
    rects = detector(gray, 0)

    # 人脸数
    for rect in rects:
        # 提取脸部68个特征点，并转换为numpy数组
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # 提取左右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 计算眼睛轮廓的凸包并绘出
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)
    # 根据视频总帧数以及视频总时长计算出每一帧的时间
    # 假如计算出时间为0.04(秒)，则waitKey()填40(ms)
    key = cv2.waitKey(40) & 0xFF
    # 为什么要加“& 0xFF”：https://blog.csdn.net/Addmana/article/details/54604298
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
