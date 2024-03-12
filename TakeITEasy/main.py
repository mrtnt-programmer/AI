import cv2
from matplotlib import pyplot as plt

img = cv2.imread('TakeITEasy\img\singlehex.png')
img = cv2.resize(img,(800,800))

plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

cv2.imshow('Coloured Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#convert to hsv

'''cv2.imshow('Coloured Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# lower range of red color in HSV
lower_range = (0, 0, 0)
# upper range of red color in HSV
upper_range = (240, 240, 240)
mask = cv2.inRange(img, lower_range, upper_range)
color_image = cv2.bitwise_and(img, img, mask=mask)
# Display the color of the image
cv2.imshow('Coloured Image', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()