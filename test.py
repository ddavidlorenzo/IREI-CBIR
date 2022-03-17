import numpy as np
import cv2
pic_path = "pokemon_dataset\\Abra\\5c0ca320656b4f2fadea7aefeb80da53.jpg"
img = cv2.imread(pic_path)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)
img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('calcHist Demo', img)
cv2.waitKey()