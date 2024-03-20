import cv2
from matplotlib import pyplot as plt

img = cv2.imread('TakeITEasy\img\singlehex.png.png')

cv2.imshow('Coloured Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# lower range of red color in HSV
lower_range = (0, 50, 50)
# upper range of red color in HSV
upper_range = (150, 255, 255)
mask = cv2.inRange(img, lower_range, upper_range)
color_image = cv2.bitwise_and(img, img, mask=mask)
# Display the color of the image
cv2.imshow('Coloured Image', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()