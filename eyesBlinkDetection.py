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
import matplotlib.pyplot as plt

# 眨眼阀值设定：当ear连续3帧小于0.25，并在下一帧大于0.25时记为一次眨眼
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# 过滤阈值以及眨眼阈值设定
FILTER = 0.04
LUOCHA_THRESH = 0.065
# 初始化连续帧数及眨眼数
COUNTER = 0
TOTAL = 0

# 用于实时绘制ear值的曲线图
ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
ear_array = []
test_ear_array = []
plt.ion()
f = 0  # 帧数
i = 0  # X轴数据
ear = 0  # Y轴数据
dizeng = 0
dijian = 0
luocha = 0
luocha_array = []
luocha_array_appended = False
closeEyes_luocha = -5  # 闭眼落差初始值选偏大负数，保证和睁眼落差相加不会满足眨眼条件
temp = 0


# 定义eye_aspect_ratio函数
def eye_aspect_ratio(eye):
    # 计算竖直方向眼睛的欧式距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平方向眼睛的欧式距离
    C = dist.euclidean(eye[0], eye[3])
    # 计算眼睛横纵比
    ear = (A + B) / (2.0 * C)
    return ear

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 获取本地视频流
source = "test/videoForTest/11.mp4"
# 获取网络视频流
# source = "http://172.28.171.45:8090/"
# 获取直播视频流
# source = "rtmp://172.28.171.74:1935/mylive/room1"
# 从本地默认设备获取视频流
# source = 0
cap = cv2.VideoCapture(source)


# 获取左右眼特征点的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# region for test
'''
for line in open("ear3.txt"):
    tmp_str = line[-8:]
    test_ear_array.append(float(tmp_str))
'''
# endregion

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
            # 提取左右眼坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # 计算眼睛轮廓的凸包并绘出
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # 左右眼平均ear值
            ear = (leftEAR + rightEAR) / 2.0

            # 四舍五入保留两位小数
            # ear = round(ear, 2)

            # Adrian Rosebrock'way to count blinks
            '''
            # ear连续3帧小于0.25时记为一次眨眼
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    COUNTER = 0
                    print("yes")
            '''
        # region for test
        # if f > len(test_ear_array):
        #     break
        # ear = test_ear_array[f-1]
        # endregion

        # 动态曲线显示ear的值
        if ear != 0:
            i += 1
            ear_array.append(ear)  # 添加 ear 到数组中
            ax.append(i)  # 添加 i 到 x 轴的数据中h
            ay.append(ear)  # 添加 ear 到 y 轴的数据中
            # print(i)
            luocha_array_appended = False

            # 过滤异常值
            ear_array_removed = False

            if len(ear_array) > 2:
                if ear_array[len(ear_array)-3] < ear_array[len(ear_array)-1]:
                    if ear_array[len(ear_array)-3] > ear_array[len(ear_array)-2]\
                            and ear_array[len(ear_array)-3] - ear_array[len(ear_array)-2] < FILTER:
                        # 拓展：删除元素3种方法
                        # remove 删除首个符合条件的元素
                        # pop 根据索引删除，会返回删除的值
                        # del 根据索引删除，不会返回删除的值
                        # 考虑到ear_array有相同值的元素，用remove可能会删错，用del最佳
                        del ear_array[len(ear_array)-2]
                        ear_array_removed = True
                        ax.remove(i-1)
                elif ear_array[len(ear_array)-3] > ear_array[len(ear_array)-1]:
                    if ear_array[len(ear_array)-2] > ear_array[len(ear_array)-3] \
                            and ear_array[len(ear_array)-2] - ear_array[len(ear_array)-3] < FILTER:
                        del ear_array[len(ear_array)-2]
                        ear_array_removed = True
                        ax.remove(i-1)
            '''
            plt.clf()  # 清除之前画的图
            plt.plot(ax, ear_array)  # x的长度即为y轴数据的个数
            plt.pause(0.1)  # 暂停0.1秒
            '''
            if len(ear_array) > 2 and not ear_array_removed:
                if ear_array[len(ear_array)-2] > ear_array[len(ear_array)-3]:
                    dizeng += 1
                    if dijian != 0:
                        luocha = ear_array[len(ear_array)-3] - ear_array[len(ear_array)-3-dijian]
                        if dijian != 1:  # 过滤递减为1的落差
                            luocha_array.append(luocha)
                            luocha_array_appended = True
                            print("-", dijian, luocha)
                        temp = dijian
                        dijian = 0

                elif ear_array[len(ear_array)-2] < ear_array[len(ear_array)-3]:
                    dijian += 1
                    if dizeng != 0:
                        luocha = ear_array[len(ear_array)-3] - ear_array[len(ear_array)-3-dizeng]
                        if dizeng != 1:
                            luocha_array.append(luocha)
                            luocha_array_appended = True
                            print("+", dizeng, luocha)
                        temp = dizeng
                        dizeng = 0

            last_ear = ear

            '''
            if len(ear_array) > 1:
                if ear > last_ear:
                    dizeng += 1
                    if dijian != 0:
                        print("-", dijian)
                        dijian = 0
                elif ear < last_ear:
                    dijian += 1
                    if dizeng != 0:
                        print("+", dizeng)
                        dizeng = 0
            last_ear = ear
            '''
        # my way to count blinks--方法一
        # 在9帧里如果中间（第5帧）正是EAR的最小值，且落差大于0.06，则判断为眨眼
        '''
        # ear_array因为前面的过滤可能长度为发生变化，此处ear_array变化（即not ear_array_removed）时才做判断
        if len(ear_array) > 9 and not ear_array_removed:
            if ear_array[len(ear_array)-6] == min(ear_array[len(ear_array)-10:len(ear_array)-1])\
                    and ear_array[len(ear_array)-10] - ear_array[len(ear_array)-6] > 0.06:
                TOTAL += 1
                print("yes")
            # print(ear_array[len(ear_array)-10:len(ear_array)-1], ear_array[len(ear_array)-6])
        '''
        # my way to count blinks--方法二
        # 当luocha_array增加了才做判断，避免重复判断
        if luocha_array_appended:
            if luocha_array[len(luocha_array)-1] < -LUOCHA_THRESH:
                closeEyes_luocha = luocha_array[len(luocha_array)-1]
                closeEyes_end = ax[len(ax)-3]  # 记录结束闭眼的所在帧
            if luocha_array[len(luocha_array)-1] > LUOCHA_THRESH:
                openEyes_luocha = luocha_array[len(luocha_array)-1]
                openEyes_start = ax[len(ax)-3-temp]  # 记录开始睁眼的所在帧
                # 判定眨眼的条件是：闭眼约等于睁眼的落差；ear在开始睁眼的值小于0.21；结束闭眼的所在帧和开始睁眼的所在帧相差小于20
                if abs(openEyes_luocha + closeEyes_luocha) < 0.1 and ear_array[len(ear_array)-3-temp] < 0.21\
                        and openEyes_start - closeEyes_end < 20:
                    TOTAL += 1
                    closeEyes_luocha = -5
                    print("yes", closeEyes_end, openEyes_start, openEyes_start - closeEyes_end)

        # 在画面上显示眨眼次数以及EAR值
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
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

plt.ioff()  # 关闭画图的窗口
cv2.destroyAllWindows()
# vs.stop()

print(ay[96:105])
print("yeyeyeye")
print(ear_array[75:84])
print(ax)
# 处理数据与原始数据对比
plt.close('all')
plt.title('Result Analysis')
plt.plot(range(1, len(ay)+1), ay, color='green', label='original data')
plt.plot(ax, ear_array, color='red', label='processed data')
plt.legend()  # 显示图例
plt.xlabel('frames')
plt.ylabel('EAR')
plt.show()


