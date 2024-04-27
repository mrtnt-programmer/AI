from functions import *
from functions_steph import *
from math import sqrt

def filter_BW(img):
  # to be done
  return img


def find_number(img, pos): 
  piece = filter_BW(-cv.cvtColor(img, cv.COLOR_BGR2GRAY))
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
      template = filter_BW(cv.imread(path + '/numbers/' + i + '.png', cv.COLOR_BGR2GRAY))
      c, w, h = template.shape[::-1]
      method = cv.TM_CCOEFF
      res = cv.matchTemplate(img,template,method)
      min_val, max_val, min_loc, coor_num = cv.minMaxLoc(res)
      crop = img[coor_num[1]: coor_num[1] + h, coor_num[0]: coor_num[0] + w]
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
          

  


