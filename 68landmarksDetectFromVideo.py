# -*- coding: utf-8 -*-

# 检测视频中的人脸68特征点

import dlib
from PIL import Image
import numpy as np
import cv2
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import threading

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

vs = FileVideoStream("test/videoForTest/video.mp4").start()  # MV.mp4
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    img_cv2 = vs.read()
    img_cv2 = imutils.resize(img_cv2, width=1000)
    # PIL处理图像
    img = Image.fromarray(img_cv2.astype('uint8')).convert('RGB')
    img_PIL = img.convert('RGBA')

    # 灰度
    img_gray = np.array(img_PIL.convert('L'))

    rects = detector(img_gray, 0)
    # 人脸数
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_cv2, rects[i]).parts()])
        img_cv2 = img_cv2.copy()

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img_cv2, pos, 2, color=(0, 255, 0))

            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_cv2, str(idx+1), pos, font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Frame", img_cv2)
    key = cv2.waitKey(10) & 0xFF
    # 为什么要加“& 0xFF”：https://blog.csdn.net/Addmana/article/details/54604298
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
