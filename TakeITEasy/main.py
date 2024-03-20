import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

image = cv2.imread('TakeITEasy\img\hex1.jpg')
image = cv2.resize(img,(800,800))

cv2.imshow('Coloured Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#convert to hsv

# lower range of red color in HSV
lower_range = (0, 0, 0)
# upper range of red color in HSV
upper_range = (240, 240, 240)
mask = cv2.inRange(image, lower_range, upper_range)
color_image = cv2.bitwise_and(image, image, mask=mask)
# Display the color of the image
cv2.imshow('Coloured Image', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

cv2.imshow('Coloured Image', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break

# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))

cv2.imshow('Coloured Image', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()