# contours -- find contours ---- edges colour or intensity

import cv2

img = cv2.imread('cv2.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Number of Contours = " + str(len(contours)))
print(len(contours[1]))

cv2.imshow('image', img)
cv2.imshow('gray image', imgray)

cv2.waitKey(0)
cv2.destroyAllWindows()
