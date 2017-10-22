# import the necessary packages
import argparse
import datetime
import time
import cv2
import numpy as np

camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("C:\\Users\\Richard\\Pictures\\Camera Roll\\WIN_20171021_14_35_55_Pro.mp4")

prev = None
composite = None
timeOfLastMotion = None

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = np.ones((5,5), np.uint8)

while True:

    original = camera.read()[1]
    original = cv2.resize(original, (640, 480), interpolation=cv2.INTER_AREA)

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

    im, contours, hierarchy = cv2.findContours(
        dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = original.copy()
    #cv2.drawContours(cnt, contours, -1, (0, 255, 0), 2)

    frameHasMotion = False
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue

        cv2.drawContours(cnt, [c], 0, (255, 0, 0), 2)

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(cnt, (x, y), (x + w, y + h), (0, 0, 255), 2)

        frameHasMotion = True
        timeOfLastMotion = time.time()

    if not frameHasMotion:
        if timeOfLastMotion is None or time.time() - timeOfLastMotion > 5:
            composite = cv2.addWeighted(composite, 0.95, current, 0.05, 0)
    composite = cv2.addWeighted(composite, 0.95, current, 0.05, 0)

    compDelta = cv2.absdiff(current, composite)
    cv2.imshow('compDelta', compDelta)
    compThresh = cv2.threshold(compDelta, 20, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('compThresh', compThresh)
    compDilate = cv2.dilate(compThresh, None, iterations=5)
    cv2.imshow('compDilate', compThresh)

    compContours = cv2.findContours(
        compDilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    compCnt = original.copy()
    cv2.drawContours(compCnt, compContours, -1, (0, 255, 0), 2)
    cv2.imshow('compCnt', compCnt)

    fgmask = fgbg.apply(original)

    erodedMask = cv2.erode(fgmask, kernel, iterations = 1)

    bg = fgbg.getBackgroundImage()
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('bg', bg)
    cv2.imshow('erodedMask', erodedMask)

    cv2.imshow('camera', original)
    cv2.imshow('current', current)
    cv2.imshow('cnt', cnt)
    cv2.imshow('composite', composite)

    cv2.waitKey(1)

    prev = current

camera.release()
cv2.destroyAllWindows()
