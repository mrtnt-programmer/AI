from functions import *
from math import sqrt

img_list = ['/perf', '/hex1', '/hex2', '/hex3', '/hex4', '/hex5', '/perf2']

# img_list = ['/perf2']

for name in img_list:
  img = cv.imread(path + name + '.jpg')
  print(path + name + '.jpg')
  affiche(img)

  img_norm = normalize_image(img)

  affiche(img_norm)


