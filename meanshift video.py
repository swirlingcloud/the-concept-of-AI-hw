import cv2
import numpy as np

# 设置初始化的窗口位置
r,h,c,w = 0,100,0,100 # 设置初试窗口位置和大小
track_window = (c,r,w,h)

cap = cv2.VideoCapture("E:\大二上\人工智能导论\opencv.mp4")

ret, frame= cap.read()
# 设置追踪的区域
roi = frame[r:r+h, c:c+w]
# roi区域的hsv图像
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 取值hsv值在(0,60,32)到(180,255,255)之间的部分
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# 归一化
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# 设置终止条件，迭代10次或者至少移动1次
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()
    if ret == True:
        # 计算每一帧的hsv图像
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 计算反向投影
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
