objects = []
springs = []

def add_object(x):
  objects.append(x)
  
def add_spring(a, b, length=None, stiffness=1):
  if length == None:
    length = ((objects[a][0] - objects[b][0]) ** 2 + (objects[a][1] - objects[b][1]) ** 2) ** 0.5
  springs.append([a, b, length, stiffness])

def robotA():
  add_object([0.2, 0.2])
  add_object([0.3, 0.2])
  add_object([0.3, 0.3])
  
  s = 4000
  add_spring(0, 1, stiffness=s)
  add_spring(0, 2, stiffness=s)
  add_spring(1, 2, stiffness=s)
  
  return objects, springs

robots = [robotA]
