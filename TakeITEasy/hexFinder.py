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
ratio = image.shape[0] / float(image.shape[0])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


show(thresh)

