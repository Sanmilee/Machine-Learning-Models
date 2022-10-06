# thresholding with images

import cv2
import numpy as np

img = cv2.imread('gradient.png')

# (source, threshold_value, value_returned, threshold_type)
_, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
_, th3 = cv2.threshold(img, 200, 0, cv2.THRESH_BINARY)

# THRESH_BINARY_INV ==> inverse of binary
_, th4 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# THRESH_TRUNC ==> all pixels above thresh_value (200) becomes 200
_, th5 = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)

# THRESH_TOZERO ==> all pixels below thresh_value (200) becomes 0 (black)
_, th6 = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)

# THRESH_TOZERO_INV ==> inverse of THRESH_TOZERO
_, th7 = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO_INV)


cv2.imshow("image", img)
cv2.imshow("th1", th1)
cv2.imshow("th2", th2)
cv2.imshow("th3", th3)
cv2.imshow("th4", th4)
cv2.imshow("th5", th5)
cv2.imshow("th6", th6)
cv2.imshow("th7", th7)


cv2.waitKey(0)
cv2.destroyAllWindows()
