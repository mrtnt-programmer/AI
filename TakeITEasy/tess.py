import pytesseract
import cv2 as cv
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
from matplotlib import pyplot as plt
import random 
import copy
import time



path = 'TakeITEasy/img'
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

name = '/hex5'
img = cv.imread(path + name + '.jpg')
print(path + name + '.jpg')
#affiche(img)

img = normalize_image(img)

# central piece:
#120 120
coors = [[None    ,None     ,[340,70] ,[455,135],[570,200]],  
        [ None    ,[220,130],[340,200],[455,265],[570,335]],
        [[105,195],[220,260],[340,340],[455,405],[570,465]],
        [[105,335],[220,400],[340,470],[455,535],None     ],
        [[105,465],[220,530],[340,600],None     ,None     ]]

size = [30,30]
"""coors = [[None    ,None     ,[377,85] ,None     ,None     ],  
        [ None    ,[220,140],[377,215],[460,140],None     ],
        [[100,200],[220,280],[377,355],[460,280],[580,200]],
        [[100,340],[220,410],[377,485],[460,410],[580,340]],
        [[100,470],[220,540],[377,615] ,[460,540],[580,470]]]
"""#left+2   left+1    #middle   #right+1  #right+2
temp = img.copy()
i = 0
for lines in coors:
  j=0
  for coor in lines:
    if coor is not None:
			#print(coor)
			#cv.rectangle(temp,coor, np.sum([coor,size],axis=0), (0,0,255), thickness=2)
      part = [80,80]
      part = np.divide((part),(100,100))
      rect0 = [int(coor[0]+(part[0]*0)),int(coor[1]+(part[1]*0)),int(coor[0]+(part[0]*100)+size[0]),int(coor[1]+part[1]*100+size[1])]
      rect1 = [int(coor[0]+(part[0]*48)),int(coor[1]+(part[1]*13)),int(coor[0]+(part[0]*52)+size[0]),int(coor[1]+part[1]*17+size[1])]
      rect2 = [int(coor[0]+(part[0]*10)),int(coor[1]+(part[1]*80)),int(coor[0]+(part[0]*20)+size[0]),int(coor[1]+part[1]*90+size[1])]
      rect3 = [int(coor[0]+(part[0]*0)),int(coor[1]+(part[1]*0)),int(coor[0]+(part[0]*100)+size[0]),int(coor[1]+part[1]*100+size[1])]
      
      cv.rectangle(temp,(rect0[0],rect0[1]),(rect0[2],rect0[3]),(0,255,255), thickness=2)#top
      cv.rectangle(temp,(rect1[0],rect1[1]),(rect1[2],rect1[3]),(255,255,0), thickness=2)#top
      cv.rectangle(temp,(rect2[0],rect2[1]),(rect2[2],rect2[3]), (255,0,0), thickness=2)#left
      cv.rectangle(temp,(rect3[0],rect3[1]),(rect3[2],rect3[3]), (255,0,0), thickness=2)#right
      coors[i][j] = [rect1,rect2,rect3]
    j += 1
  i += 1
affiche(temp)
#print(coors)

cv.imwrite(path +'/temp'+ name + '_normalized.png',temp)

print("Shape of the image", img.shape) 
piece = img.copy()
piece = piece[335:470,345:455]
#affiche(piece)
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

gray = convertGray(img,210)

# Adapted from
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html


#trio = [top,bottomLeft,bottomRight]
tuile = [[None,     None,      [[],[],[]],[[],[],[]],[[],[],[]]],#remplie de trio
        [None,      [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]],
        [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]],
        [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],None      ],
        [[[],[],[]],[[],[],[]],[[],[],[]],None      ,None      ]]


def findNumber(LocalCoor,coor):
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
   
temp = img.copy()
def initialisation():
  for line in range(len(coors)):
    for tile in range(len(coors[line])):
      if coors[line][tile] is not None:
        for coor in range(len(coors[line][tile])):
          tuile[line][tile][coor] = findNumber(coors[line][tile][coor],coor)

initialisation()
print(coors)
print(tuile)
print()

decalage = 20
def correct():
  for line in range(len(coors)):
    for tile in range(len(coors[line])):
      if coors[line][tile] is not None:
        for coor in range(len(coors[line][tile])):
          if tuile[line][tile][coor] != "":
            tuile[line][tile][coor] = tuile[line][tile][coor][0]
          else:
            localCoors = coors[line][tile][coor]
            localCoors = [localCoors[0]+random.randint(0,decalage),localCoors[1]+random.randint(0,decalage),localCoors[2]+random.randint(0,decalage),localCoors[3]+random.randint(0,decalage)]
            num = findNumber(localCoors,coor)
            if num !=  '':
              tuile[line][tile][coor] = num

for i in range(15):
   correct()
   print("loop",i)
print(coors)
print(tuile)