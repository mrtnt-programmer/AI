import cv2
from matplotlib import pyplot as plt

img = cv2.imread('TakeITEasy/img/hex1.jpg')
img = cv2.resize(img,(800,800))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#convert to hsv
"""cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

"""
plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
"""

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,th=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',th)  
cv2.waitKey(0)
cv2.destroyAllWindows()

color = ('r','g','b')
labels = ('h','s','v')
#Pour col allant r à b et pour i allant de 0 au nombre de couleurs
for i,col in enumerate(color):
    #Hist prend la valeur de l'histogramme de hsv sur la canal i.
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    # Plot de hist.
    plt.plot(hist,color = col,label=labels[i])
    plt.xlim([0,256])
plt.show()

################################################################################################


# Load image, grayscale, median blur, sharpen image
image = img
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#convert to hsv
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

# Threshold and morph close
thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours and filter using threshold area
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 100
max_area = 1500
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area and area < max_area:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        image_number += 1

cv2.imshow('sharpen', sharpen)
cv2.imshow('close', close)
cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey()

####################################################################################################


