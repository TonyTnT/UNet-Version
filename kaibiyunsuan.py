import cv2
import os
import numpy as np
import matplotlib.pylab as plt

basedir=r'C:\Users\41799\Desktop\result\result'

for filename in os.listdir(basedir):
    img = cv2.imread(os.path.join(basedir,filename), 0)
    kernel = np.ones((20, 20), np.uint8)
    openimg = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    openimg = cv2.morphologyEx(openimg,cv2.MORPH_CLOSE,kernel)
    img_gauss = cv2.GaussianBlur(openimg,(15,15),1)
    # cv2.imshow('src',img)
    # cv2.imshow('dst',openimg)
    # cv2.imshow('gau',img_gauss)
    # cv2.waitKey()
    cv2.imwrite(os.path.join(basedir,filename),img_gauss)