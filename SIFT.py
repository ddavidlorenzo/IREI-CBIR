import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

PATH_IMG1 = "pokemon_dataset\\Arbok\\2382e5e913f94dd7845e6b1ac733ef18.jpg"
PATH_IMG2 = "pokemon_dataset\\Arbok\\e2581f92a35646368e416d242d6f64ca.jpg"

PATH_IMG3 = "pokemon_dataset\\Abra\\5c0ca320656b4f2fadea7aefeb80da53.jpg"

def drawMatchesKnn(path_img1, path_img2):
    img1 = cv.imread(path_img1,cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(path_img2,cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def drawKeypoints(pic_path):
    img = cv.imread(pic_path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img = cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow('calcHist Demo', img)
    cv.waitKey()

drawMatchesKnn(PATH_IMG1, PATH_IMG2)
drawKeypoints(PATH_IMG3)