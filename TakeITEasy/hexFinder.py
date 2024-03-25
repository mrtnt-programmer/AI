import cv2 as cv
import numpy as np

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
#affiche(img)

img = normalize_image(img)

# central piece:
size = [120,120]
'''coors = [
         [100,470],[100,340],[100,200],#left+2
         [220,540],[220,410],[220,280],[220,140],#left+1
         [340,600],[340,470],[340,340],[340,200],[340,70],#middle
         [460,540],[460,410],[460,280],[460,140],#right+1
         [580,470],[580,340],[580,200]]#right+2
 '''
coors = [[None    ,None     ,[340,70] ,None     ,None     ],  
        [ None    ,[220,140],[340,200],[460,140],None     ],
        [[100,200],[220,280],[340,340],[460,280],[580,200]],
        [[100,340],[220,410],[340,470],[460,410],[580,340]],
        [[100,470],[220,540],[340,600] ,[460,540],[580,470]]]
#left+2   left+1    #middle   #right+1  #right+2
temp = img.copy()
i = 0
for lines in coors:
  j=0
  for coor in lines:
    if coor is not None:
      print(coor)
    #cv.rectangle(temp,coor, np.sum([coor,size],axis=0), (0,0,255), thickness=2)
      half = np.divide(size,[2,2])
      miniSize = np.sum([coor,half],axis=0).astype(int)
      cv.rectangle(temp,(int(coor[0]+(half[1]/2)),int(coor[1])),(int(coor[0]+(half[1]*3/2)),int(coor[1]+half[1])), (255,0,0), thickness=2)#top
      cv.rectangle(temp,(int(coor[0]),int(coor[1]+half[1])),(int(coor[0]+half[1]),int(coor[1]+size[1])), (255,0,0), thickness=2)#left
      cv.rectangle(temp,(int(coor[0]+half[0]),int(coor[1]+half[1])),np.sum([coor,size],axis=0), (255,0,0), thickness=2)#right
      coors[i][j] = [(int(coor[0]+(half[1]/2)),int(coor[1])),(int(coor[0]),int(coor[1]+half[1])),(int(coor[0]+half[0]),int(coor[1]+half[1]))]
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
            mask[i][j] = img[i][j]
  return mask 

gray = convertGray(img,210)
#affiche(gray)
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
  #print('max val:', max_val)

  top_left = max_loc
  bottom_right = (top_left[0] + w, top_left[1] + h)

  cv.rectangle(p,top_left, bottom_right, 255, 2)
  plt.subplot(121),plt.imshow(res,cmap = 'gray')
  plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(p,cmap = 'gray')
  plt.title('Detected Point'+str(digit)), plt.xticks([]), plt.yticks([])
  #plt.show() 
  return res

#trio = [top,bottomLeft,bottomRight]
tuile = [[None,None,[],None ,None  ],#remplie de trio
        [None,[]   ,[],[]    ,None ],
        [[]  ,[]   ,[],[]    ,[]   ],
        [[]  ,[]   ,[],[]    ,[]   ],
        [[]  ,[]   ,[],[]    ,[]   ]]

probability = 0

for line in range(len(coors)):
  for tile in range(len(coors[line])):
    if coors[line][tile] is not None:
      #print(len(coors[line][tile]) , len(coors[line][tile][0]))
      for coor in range(len(coors[line][tile])):
        maxscore = [7,0]#number , score of that number
        for i in range(1,10):
          piece = img.copy()
          LocalCoor = coors[line][tile][coor]
          piece = piece[LocalCoor[0]:int(LocalCoor[0]+(size[0]/2)),LocalCoor[1]:int(LocalCoor[1]+(size[1]/2))]
          #give all the pixels where numbers could be
          probability = detect(piece,i)
          min_val, max_val, min_loc, max_loc = cv.minMaxLoc(probability)
          if max_val>maxscore[1]:
            maxscore = [i,max_val]#number , score of that number
        tuile[line][tile].append(maxscore[0])
print(tuile)

#testing to find best k value       k = 6/10
def convertProbability(img,seuil):#seuil entre 0-255
  rows,cols = img.shape
  mask = np.zeros((rows,cols, 3), dtype = np.uint8)
  for i in range(rows):
      for j in range(cols):
          if img[i][j] >= seuil:
            mask[i][j] = img[i][j]
  return mask 

while True:
  for k in range(10):
    probability = detect(img,i)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(probability)
    detected = convertProbability(probability,max_val*k/10)
    print(k,max_val)
    cv.namedWindow('display', cv.WINDOW_NORMAL) 
    cv.resizeWindow('display', 900, 900) 
    cv.imshow('display', detected)    
    cv.waitKey(0) 





