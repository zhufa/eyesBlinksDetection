# -*- coding: utf-8 -*-

import cv2
import imutils
from imutils.video import FileVideoStream

# 与VideoCapture相比，FileVideoStream不支持网络视频流
fileStream = True
vs = FileVideoStream("videoForTest/57.mp4").start()
i = 0
while True:
    i += 1
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    frame = imutils.rotate(frame, 90)
    if fileStream and not vs.more():
        break
    cv2.putText(frame, "Blinks: {}".format(i), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    # 根据视频总帧数以及视频总时长计算出每一帧的时间
    # 假如计算出时间为0.04(秒)，则waitKey()填40(ms)
    key = cv2.waitKey(40) & 0xFF
    if key == ord("q"):
        break
    print(i)

'''
cap = cv2.VideoCapture('video.mp4')

frames_num=cap.get(7)  # 获取视频总帧数

print(frames_num)

while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('video',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
'''