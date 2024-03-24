import cv2 as cv
import numpy as np
import time

path = 'TakeITEasy\img'

def affiche(img):
    cv.namedWindow('display', cv.WINDOW_NORMAL) 
    cv.resizeWindow('display', 900, 900) 
    cv.imshow('display', img)
    cv.waitKey(0) 
    cv.destroyAllWindows() 
    
def find_rectangle(img):
    # Convert image to grayscale
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  
    # Convert to binary image
    _, binary_image = cv.threshold(grayscale_image, 150, 255, cv.THRESH_BINARY)
    
    # Find all the contours
    all_contours, hierarchy = cv.findContours(binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    # Loop through individual contours, keep the largest rectangle
    max = 10
    for contour in all_contours:
        if cv.contourArea(contour) > max:
            
            # Approximate contour to a polygon
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        
            # Calculate aspect ratio and bounding box
            if len(approx) == 4:
                max = cv.contourArea(contour)
                
    return approx

def normalize_image(img):
    approx = find_rectangle(img)
    pts1 = []
    for p in approx:
        pts1.append(p.tolist())
    pts1 = np.float32(pts1)    
    pts2 = np.float32([[800,0], [0,0], [0,800], [800,800]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    return cv.warpPerspective(img,M,(800,800))


name = '\hex5'
img = cv.imread(path + name + '.jpg')
affiche(img)

img = normalize_image(img)

# central piece:
temp = img.copy()
cv.rectangle(temp, (345,335), (455,470), (0,0,255), thickness=2)
affiche(temp)
cv.imwrite(path +'/temp'+ name + '_normalized.png',temp)

print("Shape of the image", img.shape) 
piece = img.copy()
piece = piece[335:470,345:455]
affiche(piece)
def convertGray(img,seuil):#seuil entre 0-255
  img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
  rows,cols = img.shape
  mask = np.zeros((rows,cols, 3), dtype = np.uint8)
  for i in range(rows):
      for j in range(cols):
          if img[i][j] >= seuil:
            mask[i][j] = img[i][j]
  return mask 

gray = convertGray(img,210)
affiche(gray)
#print(rows,cols)
"""hex1
numb = mask[79:100,8:25]#bot left
numb = mask[80:99,86:103]#bot right
numb = mask[11:31,47 :63]#top
"""

"""hex5
numb = mask[82:101 ,8:25]#bot left
numb = mask[83 :102 ,85 :101 ]#bot right
numb = mask[18:38,47:63]#top
"""
#numb = mask[21:40 ,54:65]#top hex3
#numb = gray[285 :303 ,118:133]# hex5
#numb = gray[287 :306 ,195:212]# hex5

#affiche(numb)
#cv.imwrite('8.png', numb) 

# Adapted from
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html



from matplotlib import pyplot as plt
 
# looking for a digit:
def detect(img,digit):
  connector = str(chr(92))
  template = cv.imread(path +connector+'numbers'+ connector + str(digit) + '.png', cv.IMREAD_GRAYSCALE)
  w, h = template.shape[::-1]

  method = cv.TM_CCOEFF
  p = img.copy()
  p = cv.cvtColor(p, cv.COLOR_BGR2GRAY) # does not work for colored image

  # Apply template Matching
  res = cv.matchTemplate(p,template,method)
  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
  print('max val:', max_val)

  top_left = max_loc
  bottom_right = (top_left[0] + w, top_left[1] + h)

  cv.rectangle(p,top_left, bottom_right, 255, 2)
  plt.subplot(121),plt.imshow(res,cmap = 'gray')
  plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(p,cmap = 'gray')
  plt.title('Detected Point'+str(digit)), plt.xticks([]), plt.yticks([])
  plt.show() 
  return res

def convertProbability(img,seuil):#seuil entre 0-255
  rows,cols = img.shape
  mask = np.zeros((rows,cols, 3), dtype = np.uint8)
  for i in range(rows):
      for j in range(cols):
          if img[i][j] >= seuil:
            mask[i][j] = img[i][j]
  return mask 


probability = 0
for i in range(1):
  probability = detect(img,i+1)

""" testing to find best k value       k = 6/10
cv.namedWindow('display', cv.WINDOW_NORMAL) 
cv.resizeWindow('display', 900, 900) 
while True:
  for k in range(10):
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(probability)
    detected = convertProbability(probability,max_val*k/10)
    print(k)
    cv.imshow('display', detected)    
    cv.waitKey(0) 
"""


