from functions import *

# central piece:
#120 120
# coors = [[None    ,None     ,[340,70] ,[455,135],[570,200]],
#         [ None    ,[220,130],[340,200],[455,265],[570,335]],
#         [[105,195],[220,260],[340,340],[455,405],[570,465]],
#         [[105,335],[220,400],[340,470],[455,535],None     ],
#         [[105,465],[220,530],[340,600],None     ,None     ]]
#
# size = [30,30]

def coor_calc(i,j):
    if (i,j) in [(0,0),(0,1),(1,0),(3,4),(4,3),(4,4)]:
        return None
    shift = 60
    return 402 + int((j-2)*2*shift*0.98), 129 + int((i*2+j-2) * shift *1.13)

for name in ['/perf', '/hex1', '/hex2', '/hex3', '/hex4', '/hex5']:
    img = cv.imread(path + name + '.jpg')
    print(path + name + '.jpg')
    img = normalize_image(img)
    
    temp = img.copy()

    for i in range(5):
      for j in range(5):
        coor = coor_calc(i,j)
        if coor is not None:
          cv.circle(temp,coor,8,(0,0,0),-1)

    affiche(temp,name)


