import pytesseract
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random 
import copy
import time
import platform

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

if platform.system() == 'Windows':
    path = 'TakeITEasy/img'
elif platform.system() == 'Linux':
    path = './img'
else:
    raise ValueError('Unknown OS')

def affiche(img,title='display'):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, 900, 900)
    cv.imshow(title, img)
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

def take_easy_in_top_left(img):
    img_nb = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    template = cv.imread(path + '/takeeas.jpg', cv.COLOR_BGR2GRAY)
    c, w, h = template.shape[::-1]
    method = cv.TM_CCOEFF
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv.rectangle(img_nb,top_left, bottom_right, 255, 2)
    return sum(max_loc) < 30

def normalize_image(img):
    approx = find_rectangle(img)
    pts1 = []
    for p in approx:
        pts1.append(p.tolist())
    pts1 = np.float32(pts1)    
    pts2 = np.float32([[800,0], [0,0], [0,800], [800,800]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    img_norm = cv.warpPerspective(img,M,(800,800))
    for i in range(3):
        if take_easy_in_top_left(img_norm):
            break
        (h, w) = img_norm.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv.getRotationMatrix2D((cX, cY), 90, 1.0)
        img_norm = cv.warpAffine(img_norm, M, (w, h))
    return img_norm

def convertGray(img,seuil):#seuil entre 0-255
  img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
  rows,cols = img.shape
  mask = np.zeros((rows,cols, 3), dtype = np.uint8)
  for i in range(rows):
      for j in range(cols):
          if img[i][j] >= seuil:
            mask[i][j] = 255
          else :
            mask[i][j] = 0
  return mask 

def findNumber(temp,LocalCoor,coor):
  piece = temp[LocalCoor[1]:LocalCoor[3],LocalCoor[0]:LocalCoor[2]]
  # - sign is so that the number if black with withish background:  
  piece = -cv.cvtColor(piece , cv.COLOR_BGR2GRAY)
        # if we know the position of the image on the board, only three possibilities:
  if coor == 0:
      whitelist = '159'
  if coor == 1:
      whitelist = '267'
  if coor == 2:
      whitelist = '348'
  # --psm 10 is to indicate we're looking for a single character: 
  return pytesseract.image_to_string(piece, config='--psm 10 -c tessedit_char_whitelist=' + whitelist)

# central piece:
#120 120
coors = [[None    ,None     ,[340,70] ,[455,135],[570,200]],  
        [ None    ,[220,130],[340,200],[455,265],[570,335]],
        [[105,195],[220,260],[340,340],[455,405],[570,465]],
        [[105,335],[220,400],[340,470],[455,535],None     ],
        [[105,465],[220,530],[340,600],None     ,None     ]]

size = [30,30]

def initialisation(temp,tuile):
  for line in range(len(coors)):
    for tile in range(len(coors[line])):
      if coors[line][tile] is not None:
        for coor in range(len(coors[line][tile])):
          tuile[line][tile][coor] = findNumber(temp,coors[line][tile][coor],coor)

def correct(temp,tuile,decalageD):
  decalage = decalageD*3
  nbrErreur = 0
  for line in range(len(coors)):
    for tile in range(len(coors[line])):
      if coors[line][tile] is not None:
        for coor in range(len(coors[line][tile])):
          if tuile[line][tile][coor] != "":
            tuile[line][tile][coor] = tuile[line][tile][coor][0]
          else:
            nbrErreur +=1
            localCoors = coors[line][tile][coor]
            localCoors = [localCoors[0]+random.randint(0,decalage),localCoors[1]+random.randint(0,decalage),localCoors[2]+random.randint(0,decalage),localCoors[3]+random.randint(0,decalage)]
            num = findNumber(temp,localCoors,coor)
            if num !=  '':
              tuile[line][tile][coor] = num
  return nbrErreur

