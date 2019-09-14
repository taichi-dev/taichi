objects = []
springs = []

def add_object(x, halfsize, rotation=0):
  objects.append([x, halfsize, rotation])
  
def add_spring(a, b, offset_a, offset_b, length, stiffness):
  springs.append([a, b, offset_a, offset_b, length, stiffness])

def robotA():
  add_object(x=[0.3, 0.25], halfsize=[0.15, 0.03])
  add_object(x=[0.2, 0.15], halfsize=[0.03, 0.02])
  add_object(x=[0.3, 0.15], halfsize=[0.03, 0.02])
  add_object(x=[0.4, 0.15], halfsize=[0.03, 0.02])
  add_object(x=[0.4, 0.3], halfsize=[0.005, 0.03])
  
  l = 0.12
  s = 15
  add_spring(0, 1, [-0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 1, [-0.1, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 2, [-0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 2, [0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 3, [0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 3, [0.1, 0.00], [0.0, 0.0], l, s)
  # -1 means the spring is a joint
  add_spring(0, 4, [0.1, 0], [0, -0.05], -1, s)
  
  return objects, springs


robots = [robotA]
