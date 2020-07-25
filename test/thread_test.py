# -*- coding: utf-8 -*-

# 多线程测试

from imutils.video import FileVideoStream
import time
import matplotlib.pyplot as plt
import threading

ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()

i = 0
ear = 0


def loop():
    while ear != 100:
        if ear != 0:
            ax.append(i)  # 添加 i 到 x 轴的数据中
            ay.append(ear)  # 添加 i 的平方到 y 轴的数据中
            plt.clf()  # 清除之前画的图
            plt.plot(ax,ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
            plt.pause(0.1)  # 暂停一秒


t = threading.Thread(target=loop, name='LoopThread')
t.start()
vs = FileVideoStream("videoForTest/57.mp4").start()
fileStream = True

while ear != 100:
    print 'thread %s is running...' % threading.current_thread().name
    i += 1
    ear = i*i
    time.sleep(1)

plt.ioff()  # 关闭画图的窗口
print 'thread %s ended.' % threading.current_thread().name
