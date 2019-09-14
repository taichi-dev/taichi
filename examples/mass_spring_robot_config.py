objects = []
springs = []

def add_object(x):
  objects.append(x)
  
def add_spring(a, b, length=None, stiffness=1):
  if length == None:
    length = ((objects[a][0] - objects[b][0]) ** 2 + (objects[a][1] - objects[b][1]) ** 2) ** 0.5
  springs.append([a, b, length, stiffness])

def robotA():
  add_object([0.3, 0.25])
  add_object([0.2, 0.15])
  add_object([0.4, 0.3])
  
  add_spring(0, 1)
  add_spring(0, 2)
  add_spring(1, 2)
  
  return objects, springs

robots = [robotA]
