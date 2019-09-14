objects = []
springs = []

def add_object(x):
  objects.append(x)
  
def add_spring(a, b, length=None, stiffness=1):
  if length == None:
    length = ((objects[a][0] - objects[b][0]) ** 2 + (objects[a][1] - objects[b][1]) ** 2) ** 0.5
  springs.append([a, b, length, stiffness])

def robotA():
  add_object([0.2, 0.1])
  add_object([0.3, 0.13])
  add_object([0.4, 0.1])
  add_object([0.2, 0.2])
  add_object([0.3, 0.2])
  add_object([0.4, 0.2])
  
  s = 14000
  def link(a, b):
    add_spring(a, b, stiffness=s)
    
  link(0, 1)
  link(1, 2)
  link(3, 4)
  link(4, 5)
  link(0, 3)
  link(2, 5)
  link(0, 4)
  link(1, 4)
  link(2, 4)
  
  return objects, springs

robots = [robotA]
