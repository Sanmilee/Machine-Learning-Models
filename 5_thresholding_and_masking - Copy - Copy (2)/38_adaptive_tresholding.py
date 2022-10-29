# thresholding with images

import cv2
import numpy as np

img = cv2.imread('sudoku.png', 0)

# (source, threshold_value, value_returned, threshold_type)
_, th1 = cv2.threshold(img, 48, 255, cv2.THRESH_BINARY)

# (source, max_pixel_value, adap_thresh_type, threshold_type, neighbouhood_block_size, value_of_c)
# ADAPTIVE_THRESH_MEAN_C ==> mean_of_block_neighbourod
# c ===> a contant substracted from the calculated mean

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2);
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


cv2.imshow("image", img)

cv2.imshow("th1", th1)
cv2.imshow("th2", th2)
cv2.imshow("th3", th3)

cv2.waitKey(0)
cv2.destroyAllWindows()
