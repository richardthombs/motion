import cv2
import numpy as np  

cam = cv2.VideoCapture(0)
prev = None
static = None

while True:
    original = cam.read()[1]

    current = original
    current = cv2.GaussianBlur(current,(31,31),0)
#    current = cv2.pyrMeanShiftFiltering(current, 16, 32, maxLevel=1)
#    current = cv2.pyrMeanShiftFiltering(current, 11, 51, maxLevel=0)
    current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)

    if prev is None:
        prev = current

    diff = cv2.absdiff(prev, current)
    threshold = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    #cv2.drawContours(original, contours, -1, (0,255,0), 3)

    movement = False
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #cv2.rectangle(original, (x,y),(x+w,y+h),(0,0,255),3)
        movement = True

    if static is None:
        static = original

    staticPre = cv2.GaussianBlur(static, (31,31), 0)
    staticPre = cv2.cvtColor(staticPre, cv2.COLOR_RGB2GRAY)
    staticDiff = cv2.absdiff(staticPre, current)
    staticThresh = cv2.threshold(staticDiff, 16, 255, cv2.THRESH_BINARY)[1]
    staticContours = cv2.findContours(staticThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    detect = original.copy()
    cv2.drawContours(detect, staticContours, -1, (255,0,0), 3)

    for c in staticContours:
        movement = True

    if not movement:
        static = cv2.addWeighted(static,0.5, original, 0.5, 1)

    cv2.imshow('detect', detect)
    cv2.imshow('static', static)
    cv2.imshow('staticPre', staticPre)
    cv2.imshow('staticDiff', staticDiff)

    prev = current

    if cv2.waitKey(1) == 27: 
        break  # esc to quit

cv2.destroyAllWindows()