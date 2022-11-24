# Hue, Saturation and Value (HSV) - detect blue balls
import cv2
import numpy as np

while True:
    frame = cv2.imread('smarties.png')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_b = np.array([110, 50, 50])   # lower blue limit == [110, 50, 50]
    u_b = np.array([130, 255, 255]) #  upper blue limit == [130, 255, 255]


    # pass the blue value limits to form a range
    mask = cv2.inRange(hsv, l_b, u_b)

    # apply bitwise operation on the input image named 'frame'
    # scr1, scr2, mask_function
    res = cv2.bitwise_and(frame, frame, mask=mask)


    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
