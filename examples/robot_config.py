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


def robotB():
  add_object(x=[0.3, 0.25], halfsize=[0.15, 0.03])
  add_object(x=[0.2, 0.15], halfsize=[0.03, 0.02])
  add_object(x=[0.3, 0.15], halfsize=[0.03, 0.02])
  add_object(x=[0.4, 0.15], halfsize=[0.03, 0.02])
  
  l = 0.12
  s = 15
  add_spring(0, 1, [-0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 1, [-0.1, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 2, [-0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 2, [0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 3, [0.03, 0.00], [0.0, 0.0], l, s)
  add_spring(0, 3, [0.1, 0.00], [0.0, 0.0], l, s)
  
  return objects, springs


lThighInitAng = 0
lCalfInitAng = 0
rThighInitAng = 0
rCalfInitAng = 0
initHeight = 0.15

hip_pos = [0.3, 0.5 + initHeight]

def spreadAlongHip(pos, ang):
  return [(pos[0]-hip_pos[0])*cos(ang) - (pos[1]-hip_pos[1])*sin(ang), (pos[0]-hip_pos[0])*sin(ang) + (pos[1]-hip_pos[1])*cos(ang)] + hip_pos


def robotLeg():


  #hip
  add_object(hip_pos, halfsize=[0.06, 0.11])

  #left
  add_object(x=[0.3, 0.3 + initHeight], halfsize=[0.02, 0.11])
  add_object(x=[0.3, 0.1 + initHeight], halfsize=[0.02, 0.11])
  add_object(x=[0.36, 0 + initHeight], halfsize=[0.08, 0.02])

  #right
  add_object(x=[0.3, 0.3 + initHeight], halfsize=[0.02, 0.11])
  add_object(x=[0.3, 0.1 + initHeight], halfsize=[0.02, 0.11])
  add_object(x=[0.36, 0 + initHeight], halfsize=[0.08, 0.02])

  l = 0.12
  s = 15


  add_spring(0, 1, [0, 0.1], [0.0, -0.1], 0.4, 15)
  add_spring(1, 2, [0.0, 0.1], [0.0, -0.1], 0.4, 15)
  add_spring(2, 3, [0, 0.0], [0.05, 0], 0.15, 20)

  add_spring(0, 1, [0, -0.1], [0.0, 0.1], -1, s)
  add_spring(1, 2, [0, -0.1], [0.0, 0.1], -1, s)
  add_spring(2, 3, [0, -0.1], [-0.06, 0], -1, s)
  


  add_spring(0, 4, [0, 0.1], [0.0, -0.1], 0.4, 15)
  add_spring(4, 5, [0.0, 0.1], [0.0, -0.1], 0.4, 15)
  add_spring(5, 6, [0, 0.0], [0.05, 0], 0.15, 20)

  add_spring(0, 4, [0, -0.1], [0.0, 0.1], -1, s)
  add_spring(4, 5, [0, -0.1], [0.0, 0.1], -1, s)
  add_spring(5, 6, [0, -0.1], [-0.06, 0], -1, s)


  return objects, springs


#robots = [robotA, robotB]
robots = [robotLeg]