import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img/hex1.jpg')
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


