# -*- coding: utf-8 -*-

import time
from imutils.video import VideoStream
import imutils
import cv2

source = "http://172.28.171.45:8090/"

vs = VideoStream(source).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)  # 为什么要sleep？

while True:
    # 如果这是视频流我们需要判断是否还有剩余帧数
    if fileStream and not vs.more():
        break
    # 获取视频流、重置大小、灰度化
    frame = vs.read()

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # 为什么要加“& 0xFF”：https://blog.csdn.net/Addmana/article/details/54604298
    # 按q结束
    if key == ord("q"):
        break
    # 按p暂停循环若干秒
    if key == ord("p"):
        cv2.waitKey(30000)

cv2.destroyAllWindows()
vs.stop()
