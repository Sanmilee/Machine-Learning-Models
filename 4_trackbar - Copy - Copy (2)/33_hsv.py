# Hue, Saturation and Value - HSV
import cv2
import numpy as np


cv2.namedWindow("Tracking")

while True:
    frame = cv2.imread('smarties.png')

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
