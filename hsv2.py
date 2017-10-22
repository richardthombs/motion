import numpy as np
import cv2

camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("C:\\Users\\Richard\\Pictures\\Camera Roll\\WIN_20171021_14_35_55_Pro.mp4")
camera = cv2.VideoCapture("rtsp://stony:23411Summer@192.168.1.152:88/videoMain")

prev = None
hsvComp = None
hComp = None

frameCount = 0

while(True):

    original = camera.read()[1]
    original = cv2.resize(original, (640, 480), interpolation=cv2.INTER_AREA)

    current = original.copy()
    current = cv2.GaussianBlur(current, (15, 15), 0)

    #current = cv2.pyrMeanShiftFiltering(current, 16, 32, maxLevel=1)

    hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    h = cv2.split(hsv)[0]

    if hComp is None:
        hComp = h.copy()
    else:
        hComp = cv2.addWeighted(hComp, 0.9, h, 0.1, 0)

    if prev is None:
        prev = h.copy()

    hDiff = cv2.absdiff(h, prev)
    hThresh = cv2.threshold(hDiff, 10, 255, cv2.THRESH_BINARY)[1]

    hCompDiff = cv2.absdiff(h, hComp)
    hCompThresh = cv2.threshold(hCompDiff, 10, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow('h', h)
    cv2.imshow('hDiff', hDiff)
    cv2.imshow('hComp', hComp)
    cv2.imshow('hThresh', hThresh)
    cv2.imshow('hCompDiff', hCompDiff)
    cv2.imshow('hCompThresh', hCompThresh)
    cv2.imshow('camera', original)
    cv2.imshow('hsv', hsv)

    prev = h

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
