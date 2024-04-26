from functions import *
from math import sqrt

def draw_rect(img, coor, w, h=-1, color=(255,255,255)):
    h = w if h == -1 else h
    top_left = (coor[0] - w//2, coor[1] - h//2)
    bottom_right = (coor[0] + w//2, coor[1] + h//2)
    cv.rectangle(img,top_left,bottom_right,color,thickness=2)

def coor_calc(i,j):
    if (i,j) in [(0,0),(0,1),(1,0),(3,4),(4,3),(4,4)]:
        return None
    shift = 60
    return 402 + int((j-2)*2*shift*0.98), 129 + int((i*2+j-2) * shift *1.13)

def coor_number(coor, name):
  radius = 44
  if name == 'top':
    return coor[0], coor[1]-radius
  if name == 'left':
    return int(coor[0] - radius*sqrt(3)/2), int(coor[1]+radius/2)
  if name == 'right':
    return int(coor[0] + radius*sqrt(3)/2), int(coor[1]+radius/2)



