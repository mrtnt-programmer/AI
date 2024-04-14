
import copy

coors = [[None,           None,            ['9', '7', '3'], ['9', '2', '8'], ['1', '2', '3'] ],
        [None,            ['5', '7', '8'], ['9', '6', '4'], ['1', '2', '4'], ['5', '6', '4'] ],
        [['1', '7', '8'], ['5', '2', '3'], ['9', '2', '3'], ['9', '6', '3'], ['5', '7', '3'] ],
        [['1', '6', '8'], ['5', '2', '8'], ['9', '6', '8'], ['9', '7', '8'], None            ],
        [['1', '2', '8'], ['5', '2', '4'], ['9', '7', '4'], None           , None            ]]

points = [[],[],[]]#points the line is worth

def turn3(coorz):
  newCoorz = copy.deepcopy(coorz)
  for line in range(5):
    for col in range(5):
      newCoorz[line][col] = coorz[col][line]
  return newCoorz

def turn2(coorz):
  newCoorz = [3,4,5,4,3]
  for i in range(5):
    start = (0,0)
    for col in range(4,-1,-1):
      for line in range(4,-1,-1):
        if coorz[line][col] != None:
          start = (col,line)
    K = newCoorz[i]
    newCoorz[i] = []
    if i == 0:
      newCoorz[0].append(None)
      newCoorz[0].append(None)
    if i == 1:
      newCoorz[1].append(None)
    for k in range(K):#remplis avec 
      newCoorz[i].append(coorz[start[0]+k][start[1]-k])
      coorz[start[0]+k][start[1]-k] = None
    if i == 3:
      newCoorz[3].append(None)
    if i == 4:
      newCoorz[4].append(None)
      newCoorz[4].append(None)
  return newCoorz

def getNumber(coorz,k):#count verticaly
  pointsGroup = []
  for col in range(5):
    subPointsGroup = []
    for line in range(5):
      if coorz[line][col] is not None:
        subPointsGroup.append(coorz[line][col][k])
    pointsGroup.append(subPointsGroup)
  return pointsGroup

def count(points):# verify the intire line is the same number
  score = 0
  for group in points:
    if max(group) == min(group):
      score += int(group[0])*len(group)
  return score

score = 0
#top
score += count(getNumber(coors,0))
print(score)

#right 
coors3 = turn3(copy.deepcopy(coors))
score += count(getNumber(coors3,2))
print(score)

#left 
coors2 = turn2(coors)
coors2 = turn3(coors2)
score += count(getNumber(coors2,1))
print(score)



