import cv2
import numpy as np
import math

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def distance(a,b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

#camera = cv2.VideoCapture(0)
#camera = cv2.VideoCapture("C:\\Users\\Richard\\Pictures\\Camera Roll\\WIN_20171021_14_35_55_Pro.mp4")
camera = cv2.VideoCapture("rtsp://camera:camera1234@192.168.1.152:88/videoMain")

prev = None

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:

    original = camera.read()[1]
    original = cv2.resize(original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    current = original.copy()
    #current = cv2.GaussianBlur(current, (21, 21), 0)
    #current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    mogMask = fgbg.apply(current)
    mogMask = cv2.erode(mogMask, None, iterations = 2)
    mogMask = cv2.dilate(mogMask, None, iterations = 3)
    mogContours = cv2.findContours(mogMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    mogDetect = original.copy()
    #cv2.drawContours(mogDetect, mogContours, -1, (255,255,0), 2)

    dists = []

    for c1index, c1 in enumerate(mogContours):
        m1 = cv2.moments(c1)
        x1 = int(m1["m10"] / m1["m00"])
        y1 = int(m1["m01"] / m1["m00"])

        d = []

        for c2index, c2 in enumerate(mogContours):
            m2 = cv2.moments(c2)
            x2 = int(m2["m10"] / m2["m00"])
            y2 = int(m2["m01"] / m2["m00"])
            
            dist = distance((x1,y1), (x2,y2))
            d.append(dist)

            #if dist < original.shape[0] * 0.05:
                #cv2.line(mogDetect, (x1,y1), (x2,y2), (0,0,255))

        dists.append(d)

    # One element per cluster, each element is a list of the contour numbers in the cluster
    clusters = []

    # Flag indicating if the contour has already been added to a cluster
    clusterMembership = []

    for contourNumber, _ in enumerate(mogContours):

        # Look for clusters that contain contours close to this one
        closeClusters = []
        for clusterNumber, cluster in enumerate(clusters):
            for clusteredContourNumber in cluster:
                dist = dists[contourNumber][clusteredContourNumber]
                
                if dist > 0 and dist < original.shape[0] * 0.05:
                    closeClusters.append(clusterNumber)
                    break

        if len(closeClusters) == 0:
            clusters.append([contourNumber])
        if len(closeClusters) == 1:
            clusters[closeClusters[0]].append(contourNumber)
        else:
            newCluster = [contourNumber]
            for closeCluster in closeClusters:
                for c in clusters[closeCluster]:
                    if c not in newCluster:
                        newCluster.append(c)

            for closeCluster in closeClusters:
                clusters[closeCluster]=[]

            clusters.append(newCluster)


    for clusterIndex, cluster in enumerate(clusters):
        if len(cluster) > 0:
            box = None
            for contourNumber in cluster:
                contour = mogContours[contourNumber]
                if box is None:
                    box = cv2.boundingRect(contour)
                else:
                    box = union(box, cv2.boundingRect(contour))

            cv2.rectangle(mogDetect, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)

    cv2.imshow('mogDetect', mogDetect)

    cv2.waitKey(1)

    prev = current

camera.release()
cv2.destroyAllWindows()
