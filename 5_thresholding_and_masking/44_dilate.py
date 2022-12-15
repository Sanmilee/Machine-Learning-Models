# morphological transformation - dilation

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# dilate function ==> to remove small noises
kernel = np.ones((2,2), np.uint8)
dilation = cv2.dilate(mask, kernel)

titles = ['image', 'mask', 'dilation']
images = [img, mask, dilation]

for i in range(3):
    plt.subplot(1, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
