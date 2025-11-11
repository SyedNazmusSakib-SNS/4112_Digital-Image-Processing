import cv2
import numpy as np

def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((1024,1024,channels),np.uint8)
    sh=1024/height
    sw=1024/width
    for i in range(1024):
        for j in range(1024):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage
 
img=cv2.imread("monalisa.jpg")
zoom=function(img)
cv2.imshow("nearest neighbor",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)