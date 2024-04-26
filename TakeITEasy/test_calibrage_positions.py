from functions import *
from functions_steph import *
from math import sqrt

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
        for num in ['top','left','right']:
          coor_num = coor_number(coor, num)
          draw_rect(temp,coor_num,55)

  affiche(temp,name)


