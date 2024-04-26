from functions import *
from functions_steph import *
from math import sqrt

def find_number(img, pos): 
  piece = -cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # if we know the position of the image on the board, only three possibilities:
  if pos == 'top':
      whitelist = '159'
  if pos == 'left':
      whitelist = '267'
  if pos == 'right':
      whitelist = '348'
  # --psm 10 is to indicate we're looking for a single character: 
  return pytesseract.image_to_string(piece, config='--psm 10 -c tessedit_char_whitelist=' + whitelist)

# for name in ['/perf', '/hex1', '/hex2', '/hex3', '/hex4', '/hex5']:
for name in ['/perf']:
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
          print(find_number(crop, pos))
          affiche(crop)

  


