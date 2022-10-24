# Video Boundary
import cv2

cap = cv2.VideoCapture('vtest.avi')

while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('image_frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows
