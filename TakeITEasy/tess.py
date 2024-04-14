import pytesseract
import cv2 as cv
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
from matplotlib import pyplot as plt


path = 'img'
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

name = '\hex4'
img = cv.imread(path + name + '.jpg')
#affiche(img)

img = normalize_image(img)

# central piece:
#120 120
coors = [[None    ,None     ,[340,70] ,[455,135],[570,200]],  
        [ None    ,[220,130],[340,200],[455,265],[570,335]],
        [[105,195],[220,260],[340,340],[455,405],[570,465]],
        [[105,335],[220,400],[340,470],[455,535],None     ],
        [[105,465],[220,530],[340,600],None     ,None     ]]

size = [55,55]
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
      part = np.divide((size),(100,100))
      cv.rectangle(temp,(int(coor[0]+(part[0]*90)),int(coor[1]+part[1]*20)),(int(coor[0]+(part[0]*140)),int(coor[1]+part[1]*80)), (255,255,0), thickness=2)#top
      cv.rectangle(temp,(int(coor[0]+(part[0]*20)),int(coor[1]+part[1]*130)),(int(coor[0]+(part[0]*70)),int(coor[1]+part[1]*200)), (255,0,0), thickness=2)#left
      cv.rectangle(temp,(int(coor[0]+(part[0]*160)),int(coor[1]+part[1]*130)),(int(coor[0]+(part[0]*210)),int(coor[1]+part[1]*200)), (255,0,0), thickness=2)#right
      coors[i][j] = [(int(coor[0]+(part[0]*90)),int(coor[1]+part[1]*20),int(coor[0]+(part[0]*140)),int(coor[1]+part[1]*80)),
                     (int(coor[0]+(part[0]*20)),int(coor[1]+part[1]*130),int(coor[0]+(part[0]*70)),int(coor[1]+part[1]*200)),
                     (int(coor[0]+(part[0]*160)),int(coor[1]+part[1]*130),int(coor[0]+(part[0]*210)),int(coor[1]+part[1]*200))]
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
tuile = [[None,None,[],[]  ,[]   ],#remplie de trio
        [None,[]   ,[],[]  ,[]   ],
        [[]  ,[]   ,[],[]  ,[]   ],
        [[]  ,[]   ,[],[]  ,None ],
        [[]  ,[]   ,[],None,None ]]

probability = 0

for line in range(len(coors)):
  for tile in range(len(coors[line])):
    if coors[line][tile] is not None:
      #print(len(coors[line][tile]) , len(coors[line][tile][0]))
      for coor in range(len(coors[line][tile])):
        temp = img.copy()
        LocalCoor = coors[line][tile][coor]
        piece = temp[LocalCoor[1]:LocalCoor[3],LocalCoor[0]:LocalCoor[2]]
        #give all the pixels where numbers could be
        # not sure if higher resolution 2000x2000 would be better:
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
        coors[line][tile][coor] = pytesseract.image_to_string(piece, config='--psm 10 -c tessedit_char_whitelist=' + whitelist)
        #print(whitelist,coors[line][tile][coor])########
        #affiche(piece)
        #affiche(piece)
print(coors)
#print(tuile)

for line in range(len(coors)):
  for tile in range(len(coors[line])):
    if coors[line][tile] is not None:
      for coor in range(len(coors[line][tile])):
       if coors[line][tile][coor] != "":
          coors[line][tile][coor] = coors[line][tile][coor][0]

print(coors)

         
         
          
       





