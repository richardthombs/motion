import numpy as np
import cv2

cap = cv2.VideoCapture(0)

prev = None
frameCount = 0

while(True):
    original = cap.read()[1]
    frameCount = frameCount + 1

    current=original.copy()

    Z = current.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center = cv2.kmeans(Z,K,criteria,10,flags = cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((current.shape))
    
    cv2.imshow('res2',res2)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
