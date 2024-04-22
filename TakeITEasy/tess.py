from functions import *

name = '/hex5'
img = cv.imread(path + name + '.jpg')
print(path + name + '.jpg')
#affiche(img)

img = normalize_image(img)


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
      part = [90,90]
      part = np.divide((part),(100,100))
      rect0 = [int(coor[0]+(part[0]*0)),int(coor[1]+(part[1]*0)),int(coor[0]+(part[0]*100)+size[0]),int(coor[1]+part[1]*100+size[1])]
      rect1 = [int(coor[0]+(part[0]*48)),int(coor[1]+(part[1]*13)),int(coor[0]+(part[0]*52)+size[0]),int(coor[1]+part[1]*17+size[1])]
      rect2 = [int(coor[0]+(part[0]*10)),int(coor[1]+(part[1]*80)),int(coor[0]+(part[0]*15)+size[0]),int(coor[1]+part[1]*90+size[1])]
      rect3 = [int(coor[0]+(part[0]*85)),int(coor[1]+(part[1]*80)),int(coor[0]+(part[0]*90)+size[0]),int(coor[1]+part[1]*90+size[1])]
      
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

gray = convertGray(img,210)

# Adapted from
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html


#trio = [top,bottomLeft,bottomRight]
tuile = [[None,     None,      [[],[],[]],[[],[],[]],[[],[],[]]],#remplie de trio
        [None,      [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]],
        [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]],
        [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],None      ],
        [[[],[],[]],[[],[],[]],[[],[],[]],None      ,None      ]]

   
temp = img.copy()
initialisation(temp,tuile)
print(coors)
print(tuile)
print()

decalage = 30

for i in range(15):
   print("loop: ",i,"erreur :",correct(temp,tuile,i))
print(coors)
print(tuile)
