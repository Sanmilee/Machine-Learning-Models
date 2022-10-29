# thresholding with images
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gradient.png', 0)

# (source, threshold_value, value_returned, threshold_type)
_, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
_, th3 = cv2.threshold(img, 200, 0, cv2.THRESH_BINARY)
_, th4 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
_, th5 = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
_, th6 = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)


titles = ["Original", 'Binary', 'Binary2', 'Binary_Inv', 'Trunc', '2_Zero']
images = [img, th1, th3, th4, th5, th6]


for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')  # plt.subplt(row, col, ind)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
