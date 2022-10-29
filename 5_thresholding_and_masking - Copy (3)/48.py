# morphological transformation - dilation
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
#_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)


# dilate function ==> to remove small noises (you may increase the window)
kernel = np.ones((2,2), np.uint8)

dilation = cv2.dilate(img, kernel, iterations=1)
erosion = cv2.erode(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
mg = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


titles = ['image', 'new_img', 'dilation', 'erosion', 'opening', 'clsoing', 'mg', 'th']
images = [img, img, dilation, erosion, opening, closing, mg, th]

for i in range(8):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
