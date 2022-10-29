# add switch using trackbar
# trackbar - used to change values dynamically in an image at runtime
import cv2
import numpy as np

cv2.namedWindow('image') # create a window with the name "image"

def nothing(x):   # takes in the trackbar value received (bet 0-255)
    print(x)

switch = "Color/Gray"   # string -- display info
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

# create trackbar('name_of_trackbar', 'frame_name_to_implemnt_trackbar', min_value, max_value, callBack_function)
cv2.createTrackbar('CP', 'image', 10, 400, nothing)   # 1st trackbar created


while(1):
    img = cv2.imread('lena.jpg')

    # implemnt trackbar - get the trackbar value ('name_of_tracker', 'frame_of_trackbar')
    s = cv2.getTrackbarPos(switch, 'image')
    pos = cv2.getTrackbarPos('CP', 'image')

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(pos), (90, 130), font, 5, (0, 0, 255), 5)


    # implement switch
    if s == 1:
        # set the gotten trackbar value into the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert to gray image

    else:
        pass # do nothing

    cv2.imshow('image', img)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

cv2.destroyAllWindows()
