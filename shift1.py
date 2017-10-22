import numpy as np
import cv2

cap = cv2.VideoCapture(0)

composite = None
prev = None
frameCount = 0
hComposite = None

while(True):
    original = cap.read()[1]
    frameCount = frameCount + 1

    current=original.copy()

    #current = cv2.resize(current, (640, 480), interpolation=cv2.INTER_AREA)
    #current = cv2.pyrMeanShiftFiltering(current, 16, 48, maxLevel=0)

    if composite is None:
        composite = current.copy()
    else:
        composite = cv2.addWeighted(composite,0.5, current, 0.5, 1)

    hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    h = cv2.split(hsv)[2]

    if hComposite is None:
        hComposite = h.copy()
    else:
        hComposite = cv2.addWeighted(hComposite, 0.9, h, 0.1, 1)

    if prev is None:
        prev = h.copy()

    hDiff = cv2.absdiff(h, prev)

    hThresh = cv2.threshold(hDiff, 16, 255, cv2.THRESH_BINARY)[1]

    compHDiff = cv2.absdiff(h, hComposite)

    cv2.imshow('current', current)
    cv2.imshow('composite', composite)
    cv2.imshow('h', h)
    cv2.imshow('hDiff', hDiff)
    cv2.imshow('camera', original)
    cv2.imshow('hThresh', hThresh)
    cv2.imshow('compHDiff', compHDiff)
    cv2.imshow('hComposite', hComposite)

    prev = h

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
