from functions import *
from functions_steph import *
from math import sqrt

def filter_BW(img):
  affiche(img)
  """#img = cv.medianBlur(img, 5)
  #show(img)

  #sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
  #img = cv.filter2D(img, -1, sharpen_kernel)
  #affiche(img)"""

  #160 too weak (the gray background of a 2 overlaps) 190 too strong (some pixels inside a 1 don't pick up) 
  filterStrength = 185

  img = cv.threshold(img, filterStrength, 255, cv.THRESH_BINARY_INV)[1]
  affiche(img)

  rows,cols,c = img.shape
  pureImage = img
  for i in range(rows):
      for j in range(cols):
          test = (img[i][j] != [0])
          #print(img[i][j],test,test[0] and test[1] and test[2])
          if test[0] or test[1] or test[2]:
             pureImage[i][j] = [255,255,255]
  affiche(pureImage)
  """
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  affiche(img)           
  img = cv.threshold(img, 160, 255, cv.THRESH_BINARY)[1]
  affiche(img) """  

  return pureImage


def find_number(img, pos): 
  piece = filter_BW(img)
        # if we know the position of the image on the board, only three possibilities:
  if pos == 'top':
      whitelist = '159'
  if pos == 'left':
      whitelist = '267'
  if pos == 'right':
      whitelist = '348'
  # --psm 10 is to indicate we're looking for a single character: 
  result = pytesseract.image_to_string(piece, config='--psm 10 -c tessedit_char_whitelist=' + whitelist)
  if result == '':
    for i in whitelist:
      template = cv.imread(path + '/numbers/' + i + '.png', cv.COLOR_BGR2GRAY)
      c, w, h = template.shape[::-1]
      method = cv.TM_CCOEFF
      res = cv.matchTemplate(img,template,method)
      min_val, max_val, min_loc, coor_num = cv.minMaxLoc(res)
      crop = img[coor_num[1]: coor_num[1] + h, coor_num[0]: coor_num[0] + w]
      affiche(crop)
      crop = filter_BW(crop)
      affiche(crop)
      result = pytesseract.image_to_string(crop, config='--psm 10 -c tessedit_char_whitelist=' + whitelist)
      if result != '':
        break
  
  print('result:', result)
  affiche(piece)
  return result 

# for name in ['/perf', '/hex1', '/hex2', '/hex3', '/hex4', '/hex5']:
for name in ['/hex1']:
  img = cv.imread(path + name + '.jpg')
  print(path + name + '.jpg')
  img = normalize_image(img)

  temp = img.copy()

  for i in range(5):
    for j in range(5):
      coor = coor_calc(i,j)
      if coor is not None:
        cv.circle(temp,coor,8,(0,0,0),-1)
        for pos in ['top','left','right']:
          coor_num = coor_number(coor, pos)
          h = 55
          w = 55
          crop = img[coor_num[1] - h//2 : coor_num[1] + h//2, coor_num[0] - w//2: coor_num[0] + w//2]
          num = find_number(crop, pos)
          

  


