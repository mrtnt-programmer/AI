import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('TakeITEasy\img\hexSimple.jpg')
#img = cv2.resize(img,(800,800))

def show(image):
  cv2.imshow('Coloured Image', image) 
  cv2.waitKey(0)
  cv2.destroyAllWindows()

show(img)

image = img
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#convert to hsv
show(image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show(gray)
blur = cv2.medianBlur(gray, 5)
show(blur)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
show(sharpen)
thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


show(close)


print(close.shape , img.shape)
rows,cols = close.shape
coloredImage = img
for i in range(rows):
    for j in range(cols):
        if close[i][j] != 0:
          coloredImage[i][j] = close[i][j]

show(coloredImage)




