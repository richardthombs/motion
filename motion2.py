# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2

camera = cv2.VideoCapture(0)
time.sleep(0.25)

lastFrame = None
composite = None
timeOfLastMotion = None

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:

    grabbed, original = camera.read()
    thisFrame = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    thisFrame = cv2.GaussianBlur(thisFrame, (21, 21), 0)

    #thisFrame = imutils.resize(thisFrame, width=500)

    if lastFrame is None:
        lastFrame = thisFrame.copy()

    if composite is None:
        composite = thisFrame.copy()

    delta = cv2.absdiff(thisFrame, lastFrame)
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
            composite = cv2.addWeighted(composite, 0.95, thisFrame, 0.05, 0)

    compDelta = cv2.absdiff(thisFrame, composite)
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
    bg = fgbg.getBackgroundImage()
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('bg', bg)

    cv2.imshow('this', thisFrame)
    #cv2.imshow('delta', delta)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('dialate', dilate)
    cv2.imshow('cnt', cnt)
    cv2.imshow('composite', composite)

    cv2.waitKey(100)

    lastFrame = thisFrame.copy()

camera.release()
cv2.destroyAllWindows()
