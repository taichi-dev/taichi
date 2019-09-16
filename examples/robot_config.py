import math

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


l_thigh_init_ang = 10
l_calf_init_ang = -10
r_thigh_init_ang = 10
r_calf_init_ang = -10
initHeight = 0.15

hip_pos = [0.3, 0.5 + initHeight]
thigh_half_length = 0.11
calf_half_length = 0.11

foot_half_length = 0.08


def rotAlong(half_length, deg, center):
  ang = math.radians(deg)
  return [half_length*math.sin(ang) + center[0], -half_length*math.cos(ang) + center[1]] 


half_hip_length = 0.08

def robotLeg():
  #hip
  add_object(hip_pos, halfsize=[0.06, half_hip_length])
  hip_end = [hip_pos[0], hip_pos[1] - (half_hip_length-0.01)]

  #left  
  l_thigh_center = rotAlong(thigh_half_length, l_thigh_init_ang, hip_end)
  l_thigh_end = rotAlong(thigh_half_length * 2.0, l_thigh_init_ang, hip_end)
  add_object(l_thigh_center, halfsize=[0.02, thigh_half_length], rotation=math.radians(l_thigh_init_ang))
  add_object(rotAlong(calf_half_length, l_calf_init_ang, l_thigh_end), halfsize=[0.02, calf_half_length], rotation = math.radians(l_calf_init_ang))
  l_calf_end = rotAlong(2.0 * calf_half_length, l_calf_init_ang, l_thigh_end)
  add_object([l_calf_end[0] + foot_half_length, l_calf_end[1]], halfsize=[foot_half_length, 0.02])

  #right 
  add_object(rotAlong(thigh_half_length, r_thigh_init_ang, hip_end), halfsize=[0.02, thigh_half_length], rotation=math.radians(r_thigh_init_ang))
  r_thigh_end = rotAlong(thigh_half_length * 2.0, r_thigh_init_ang, hip_end)
  add_object(rotAlong(calf_half_length, r_calf_init_ang, r_thigh_end), halfsize=[0.02, calf_half_length], rotation = math.radians(r_calf_init_ang))
  r_calf_end = rotAlong(2.0 * calf_half_length, r_calf_init_ang, r_thigh_end)
  add_object([r_calf_end[0] + foot_half_length, r_calf_end[1]], halfsize=[foot_half_length, 0.02])

  s = 200

  thigh_relax = 0.9
  leg_relax = 0.9
  foot_relax = 0.7

  thigh_stiff = 5
  leg_stiff = 20
  foot_stiff = 40

  #left springs
  add_spring(0, 1, [0,(half_hip_length-0.01)*0.4], [0,-thigh_half_length], thigh_relax*(2.0*thigh_half_length + 0.22), thigh_stiff)
  add_spring(1, 2, [0,thigh_half_length], [0,-thigh_half_length], leg_relax * 4.0* thigh_half_length, leg_stiff)
  add_spring(2, 3, [0,0], [foot_half_length,0], foot_relax*math.sqrt(pow(thigh_half_length,2)+pow(2.0*foot_half_length,2)), foot_stiff)

  add_spring(0, 1, [0, -(half_hip_length-0.01)], [0.0, thigh_half_length], -1, s)
  add_spring(1, 2, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
  add_spring(2, 3, [0, -thigh_half_length], [-foot_half_length,0], -1, s)

  #right springs
  add_spring(0, 4, [0,(half_hip_length-0.01)*0.4], [0,-thigh_half_length], thigh_relax*(2.0*thigh_half_length + 0.22), thigh_stiff)
  add_spring(4, 5, [0,thigh_half_length], [0,-thigh_half_length], leg_relax * 4.0* thigh_half_length, leg_stiff)
  add_spring(5, 6, [0,0], [foot_half_length,0], foot_relax*math.sqrt(pow(thigh_half_length,2)+pow(2.0*foot_half_length,2)), foot_stiff)

  add_spring(0, 4, [0, -(half_hip_length-0.01)], [0.0, thigh_half_length], -1, s)
  add_spring(4, 5, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
  add_spring(5, 6, [0, -thigh_half_length], [-foot_half_length,0], -1, s)



  return objects, springs


#robots = [robotA, robotB]
robots = [robotLeg]
