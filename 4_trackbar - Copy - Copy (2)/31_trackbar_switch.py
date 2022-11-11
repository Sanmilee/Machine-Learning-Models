# add switch using trackbar
# trackbar - used to change values dynamically in an image at runtime
import cv2
import numpy as np

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image') # create a window with the name "image"

def nothing(x):   # takes in the trackbar value received (bet 0-255)
    print(x)

switch = "0 : OFF \n 1: ON"   # string -- display info
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

# create trackbar('name_of_trackbar', 'frame_name_to_implemnt_trackbar', min_value, max_value, callBack_function)
cv2.createTrackbar('B', 'image', 0, 255, nothing)   # 1st trackbar created
cv2.createTrackbar('G', 'image', 0, 255, nothing)   # 2nd trackbar created
cv2.createTrackbar('R', 'image', 0, 255, nothing)   # 3rd trackbar created

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1)

    if k == 27:
        break


    # implemnt trackbar - get the trackbar value ('name_of_tracker', 'frame_of_trackbar')
    s = cv2.getTrackbarPos(switch, 'image')

    b = cv2.getTrackbarPos('B', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    r = cv2.getTrackbarPos('R', 'image')


    # implement switch
    if s == 1:
        # set the gotten trackbar value into the image
        img[:] = [b, g, r]


    else:
        img[:] = 0 # do nothing --  no change/variation in color


cv2.destroyAllWindows()
