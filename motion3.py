# import the necessary packages
import argparse
import datetime
import time
import cv2
import numpy as np

def removeSmallContours(contours, minArea):
    array = []
    for c in contours:
        if cv2.contourArea(c) >= minArea:
            array.append(c)
    return array

camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("C:\\Users\\Richard\\Pictures\\Camera Roll\\WIN_20171021_14_35_55_Pro.mp4")
camera = cv2.VideoCapture("rtsp://stony:23411Summer@192.168.1.152:88/videoMain")
prev = None
composite = None
timeOfLastMotion = None
movementMask = None

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:

    original = camera.read()[1]
    original = cv2.resize(original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    current = original.copy()
    #current = cv2.GaussianBlur(current, (21, 21), 0)
    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    if prev is None:
        prev = current.copy()

    if composite is None:
        composite = current.copy()

    delta = cv2.absdiff(current, prev)
    thresh = cv2.threshold(delta, 3, 255, cv2.THRESH_BINARY)[1]
    dilate = cv2.dilate(thresh, None, iterations=1)

    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    cnt = original.copy()
    cv2.drawContours(cnt, contours, -1, (0, 255, 0), 2)

    composite = cv2.addWeighted(composite, 0.95, current, 0.05, 0)

    compDelta = cv2.absdiff(current, composite)
    compMask = cv2.threshold(compDelta, 10, 255, cv2.THRESH_BINARY)[1]
    compMask = cv2.erode(compMask, None, iterations = 1)
    compContours = cv2.findContours(compMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    #compContours = removeSmallContours(compContours, 10)

    compDetect = original.copy()
    cv2.drawContours(compDetect, compContours, -1, (0, 255, 0), 2)
    #cv2.imshow('compMask', compMask)
    #cv2.imshow('compDetect', compDetect)

    mogMask = fgbg.apply(original)
    mogMask = cv2.erode(mogMask, None, iterations = 2)
    mogMask = cv2.dilate(mogMask, None, iterations = 5)
    mogContours = cv2.findContours(mogMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    mogDetect = original.copy()
    cv2.drawContours(mogDetect, mogContours, -1, (255,255,0), 2)

    #cv2.imshow('mogMask', mogMask)
    cv2.imshow('mogDetect', mogDetect)

    #cv2.imshow('camera', original)

    if movementMask is None:
        movementMask = np.zeros((original.shape[0], original.shape[1]), np.uint8)

    movementMask = cv2.bitwise_or(movementMask, mogMask)
    movementMask = cv2.add(movementMask, -1)
    #cv2.imshow('movementMask', movementMask)

    cv2.waitKey(1)

    prev = current

camera.release()
cv2.destroyAllWindows()
