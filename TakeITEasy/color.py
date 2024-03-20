import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('TakeITEasy\img\hexSimple.jpg')
img = cv2.resize(img,(800,800))

cv2.imshow('Coloured Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = img
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#convert to hsv
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


cv2.imshow('Coloured Image', close) 
cv2.waitKey(0)
cv2.destroyAllWindows()

print(close.shape , img.shape)
rows,cols = close.shape
coloredImage = close
for i in range(rows):
    for j in range(cols):
        if close[i,j] == 0:
          coloredImage[i,j] = 0

cv2.imshow('Coloured Image', coloredImage) 
cv2.waitKey(0)
cv2.destroyAllWindows()



