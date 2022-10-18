# histograms - display imaghe hist using matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', 0)

cv2.imshow("Image", img)

plt.hist(img.ravel(), 256, [0, 256])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
