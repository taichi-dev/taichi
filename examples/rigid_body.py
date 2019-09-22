from robot_config import robots
import sys

import taichi_lang as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from renderer_vector import VectorRenderer
import time
from matplotlib.pyplot import cm

renderer = VectorRenderer()

real = ti.f32
ti.set_default_fp(real)

max_steps = 4096
vis_interval = 256
output_vis_interval = 16
steps = 2048
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

use_toi = False

# ti.cfg.arch = ti.cuda

x = vec()
v = vec()
rotation = scalar()
# angular velocity
omega = scalar()

halfsize = vec()

inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
x_inc = vec()
rotation_inc = scalar()
omega_inc = scalar()

head_id = 3
goal = [0.9, 0.15]

n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -9.8
friction = 1.0
penalty = 1e4
damping = 10

gradient_clip = 30
spring_omega = 30
default_actuation = 0.05

n_springs = 0
spring_anchor_a = ti.global_var(ti.i32)
spring_anchor_b = ti.global_var(ti.i32)
# spring_length = -1 means it is a joint
spring_length = scalar()
spring_offset_a = vec()
spring_offset_b = vec()
spring_phase = scalar()
spring_actuation = scalar()
spring_stiffness = scalar()

n_sin_waves = 10
weights = scalar()
bias = scalar()


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, rotation, rotation_inc,
                                                              omega, v_inc, x_inc,
                                                              omega_inc)
  ti.root.dense(ti.i, n_objects).place(halfsize, inverse_mass, inverse_inertia)
  ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                       spring_length, spring_offset_a,
                                       spring_offset_b, spring_stiffness,
                                       spring_phase, spring_actuation)
  ti.root.dense(ti.ij, (n_springs, n_sin_waves)).place(weights)
  ti.root.dense(ti.i, n_springs).place(bias)
  ti.root.place(loss)
  ti.root.lazy_grad()


dt = 0.001
learning_rate = 1.0


@ti.func
def rotation_matrix(r):
  return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.kernel
def initialize_properties():
  for i in range(n_objects):
    inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
    inverse_inertia[i] = 1.0 / (4 / 3 * halfsize[i][0] * halfsize[i][1] * (
        halfsize[i][0] * halfsize[i][0] + halfsize[i][1] * halfsize[i][1]))
    # ti.print(inverse_mass[i])
    # ti.print(inverse_inertia[i])


@ti.func
def cross(a, b):
  return a[0] * b[1] - a[1] * b[0]


@ti.func
def to_world(t, i, rela_x):
  rot = rotation[t, i]
  rot_matrix = rotation_matrix(rot)
  
  rela_pos = rot_matrix @ rela_x
  rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])
  
  world_x = x[t, i] + rela_pos
  world_v = v[t, i] + rela_v
  
  return world_x, world_v, rela_pos


@ti.func
def apply_impulse(t, i, impulse, location, toi_input):
  # ti.print(toi)
  delta_v = impulse * inverse_mass[i]
  delta_omega = cross(location - x[t, i], impulse) * inverse_inertia[i]

  toi = ti.min(ti.max(0.0, toi_input), dt)

  ti.atomic_add(x_inc[t + 1, i], toi * (-delta_v))
  ti.atomic_add(rotation_inc[t + 1, i], toi * (-delta_omega))

  ti.atomic_add(v_inc[t + 1, i], delta_v)
  ti.atomic_add(omega_inc[t + 1, i], delta_omega)


@ti.kernel
def collide(t: ti.i32):
  for i in range(n_objects):
    hs = halfsize[i]
    for k in ti.static(range(4)):
      # the corner for collision detection
      offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1])
      
      corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
      corner_v = corner_v + dt * gravity * ti.Vector([0.0, 1.0])
      
      # Apply impulse so that there's no sinking
      normal = ti.Vector([0.0, 1.0])
      tao = ti.Vector([1.0, 0.0])
      
      rn = cross(rela_pos, normal)
      rt = cross(rela_pos, tao)
      impulse_contribution = inverse_mass[i] + ti.sqr(rn) * \
                             inverse_inertia[i]
      timpulse_contribution = inverse_mass[i] + ti.sqr(rt) * \
                              inverse_inertia[i]
      
      rela_v_ground = normal.dot(corner_v)
      
      impulse = 0.0
      timpulse = 0.0
      new_corner_x = corner_x + dt * corner_v
      toi = 0.0
      if rela_v_ground < 0 and new_corner_x[1] < ground_height:
        impulse = -(1 + elasticity) * rela_v_ground / impulse_contribution
        if impulse > 0:
          # friction
          timpulse = -corner_v.dot(tao) / timpulse_contribution
          timpulse = ti.min(friction * impulse,
                            ti.max(-friction * impulse, timpulse))
          if corner_x[1] > ground_height:
            toi = -(corner_x[1] - ground_height) / ti.min(corner_v[1], 1e-3)

      apply_impulse(t, i, impulse * normal + timpulse * tao, new_corner_x, toi)

      penalty = 0.0
      if new_corner_x[1] < ground_height:
        # apply penalty
        penalty = -dt * penalty * (
            new_corner_x[1] - ground_height) / impulse_contribution

      apply_impulse(t, i, penalty * normal, new_corner_x, 0)


@ti.kernel
def apply_spring_force(t: ti.i32):
  for i in range(n_springs):
    a = spring_anchor_a[i]
    b = spring_anchor_b[i]
    pos_a, vel_a, rela_a = to_world(t, a, spring_offset_a[i])
    pos_b, vel_b, rela_b = to_world(t, b, spring_offset_b[i])
    dist = pos_a - pos_b
    length = dist.norm() + 1e-4
    
    actuation = 0.0
    for j in ti.static(range(n_sin_waves)):
      actuation += weights[i, j] * ti.sin(
        spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)
    actuation += bias[i]
    actuation = ti.tanh(actuation)
    
    is_joint = spring_length[i] == -1
    
    target_length = spring_length[i] * (1.0 + spring_actuation[i] * actuation)
    if is_joint:
      target_length = 0.0
    impulse = dt * (length - target_length) * spring_stiffness[
      i] / length * dist
    
    if is_joint:
      rela_vel = vel_a - vel_b
      rela_vel_norm = rela_vel.norm() + 1e-1
      impulse_dir = rela_vel / rela_vel_norm
      impulse_contribution = inverse_mass[a] + ti.sqr(
        cross(impulse_dir, rela_a)) * inverse_inertia[
                               a] + inverse_mass[b] + ti.sqr(cross(impulse_dir,
                                                                   rela_b)) * \
                             inverse_inertia[
                               b]
      # project relative velocity
      impulse += rela_vel_norm / impulse_contribution * impulse_dir
    
    apply_impulse(t, a, -impulse, pos_a, 0.0)
    apply_impulse(t, b, impulse, pos_b, 0.0)


@ti.kernel
def advance_toi(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * damping)
    v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
      [0.0, 1.0])
    x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]
    omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
    rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i] + rotation_inc[t, i]
    
@ti.kernel
def advance_no_toi(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * damping)
    v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
      [0.0, 1.0])
    x[t, i] = x[t - 1, i] + dt * v[t, i]
    omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
    rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
  loss[None] = (x[t, head_id] - ti.Vector(goal)).norm()
  # loss[None] = 10 * ti.abs(x[t, head_id][0] - goal[0]) + ti.abs(x[t, head_id][1] - goal[1])

import taichi as tc
gui = tc.core.GUI('Rigid Body Simulation', tc.Vectori(1024, 1024))
canvas = gui.get_canvas()

def forward(output=None, visualize=True):
  initialize_properties()

  interval = vis_interval
  total_steps = steps
  if output:
    interval = output_vis_interval
    os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
    total_steps *= 2

  for t in range(1, total_steps):
    collide(t - 1)
    apply_spring_force(t - 1)
    if use_toi:
      advance_toi(t)
    else:
      advance_no_toi(t)
    
    if (t + 1) % interval == 0 and visualize:
      canvas.clear(0xFFFFFF)

      for i in range(n_objects):
        points = []
        for k in range(4):
          offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
          rot = rotation[t, i]
          rot_matrix = np.array(
            [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
          
          pos = np.array(
            [x[t, i][0], x[t, i][1]]) + offset_scale * rot_matrix @ np.array(
            [halfsize[i][0], halfsize[i][1]])

          points.append((pos[0], pos[1]))

        for k in range(4):
          canvas.path(tc.Vector(*points[k]), tc.Vector(*points[(k + 1) % 4])).color(0x0).radius(2).finish()
        '''
        if (i == 0):
          renderer.draw_dot([x[t, i][0], x[t, i][1]],5000, cmap(0),20)
        elif (i == 1 or i == 4):
          renderer.draw_polygon(points, cmap(0.1))
        elif (i == 2 or i == 5):
          renderer.draw_polygon(points, cmap(0.3))
        elif (i == 3 or i == 6):
          renderer.draw_polygon(points, cmap(0.7))
        '''

      # renderer.draw_dot([x[t, head_id][0], x[t, head_id][1]],color=cmap(0.6),layer=20,ec='r')
      # renderer.draw_dot([goal[0], goal[1]],layer=20,ec='r')

      for i in range(n_springs):
        def get_world_loc(i, offset):
          rot = rotation[t, i]
          rot_matrix = np.array(
            [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
          pos = np.array(
            [[x[t, i][0]], [x[t, i][1]]]) + rot_matrix @ np.array(
            [[offset[0]], [offset[1]]])
          return pos
        
        pt1 = get_world_loc(spring_anchor_a[i], spring_offset_a[i])
        pt2 = get_world_loc(spring_anchor_b[i], spring_offset_b[i])

        if spring_length[i] == -1:
          canvas.path(tc.Vector(*pt1), tc.Vector(*pt2)).color(0x000000).radius(9).finish()
          canvas.path(tc.Vector(*pt1), tc.Vector(*pt2)).color(0xFF2233).radius(7).finish()
        else:
          canvas.path(tc.Vector(*pt1), tc.Vector(*pt2)).color(0x000000).radius(7).finish()
          canvas.path(tc.Vector(*pt1), tc.Vector(*pt2)).color(0xFBCCAA).radius(5).finish()

      canvas.path(tc.Vector(0.05, ground_height), tc.Vector(0.95, ground_height)).color(0x0).radius(5).finish()

      gui.update()

  loss[None] = 0
  compute_loss(steps - 1)


@ti.kernel
def clear_states():
  for t in range(0, max_steps):
    for i in range(0, n_objects):
      v_inc[t, i] = ti.Vector([0.0, 0.0])
      x_inc[t, i] = ti.Vector([0.0, 0.0])
      rotation_inc[t, i] = 0.0
      omega_inc[t, i] = 0.0
      

def setup_robot(objects, springs, h_id):
  global head_id
  head_id = h_id
  global n_objects, n_springs
  n_objects = len(objects)
  n_springs = len(springs)
  
  print('n_objects=', n_objects, '   n_springs=', n_springs)
  
  for i in range(n_objects):
    x[0, i] = objects[i][0]
    halfsize[i] = objects[i][1]
    rotation[0, i] = objects[i][2]
  
  for i in range(n_springs):
    s = springs[i]
    spring_anchor_a[i] = s[0]
    spring_anchor_b[i] = s[1]
    spring_offset_a[i] = s[2]
    spring_offset_b[i] = s[3]
    spring_length[i] = s[4]
    spring_stiffness[i] = s[5]
    if s[6]:
      spring_actuation[i] = s[6]
    else:
      spring_actuation[i] = default_actuation
      
def optimize(toi=True, visualize=True):
  global use_toi
  use_toi = toi
  for i in range(n_springs):
    for j in range(n_sin_waves):
      weights[i, j] = np.random.randn() * 0.1
  
  losses = []
  for iter in range(20):
    clear_states()
    
    with ti.Tape(loss):
      forward(visualize=visualize)
    
    print('Iter=', iter, 'Loss=', loss[None])
    
    total_norm_sqr = 0
    for i in range(n_springs):
      for j in range(n_sin_waves):
        total_norm_sqr += weights.grad[i, j] ** 2
      total_norm_sqr += bias.grad[i] ** 2
    
    print(total_norm_sqr)
    
    norm = total_norm_sqr ** 0.5
    scale = learning_rate * min(1.0, gradient_clip / (norm + 1e-4))
    if norm > 1e3:
      continue
    for i in range(n_springs):
      for j in range(n_sin_waves):
        weights[i, j] -= scale * weights.grad[i, j]
      bias[i] -= scale * bias.grad[i]
    losses.append(loss[None])
  return losses
  

def main():
  robot_id = 0
  if len(sys.argv) != 3:
    print("Usage: python3 rigid_body.py [robot_id=0, 1, 2, ...] cmd")
  else:
    robot_id = int(sys.argv[1])
    cmd = sys.argv[2]
  print(robot_id, cmd)
  setup_robot(*robots[robot_id]())
  
  if cmd == 'plot':
    ret = {}
    for toi in [False, True]:
      ret[toi] = []
      for i in range(5):
        losses = optimize(toi=toi, visualize=False)
        # losses = gaussian_filter(losses, sigma=3)
        ret[toi].append(losses)
  
    import pickle
    pickle.dump(ret, open('losses.pkl', 'wb'))
    print("Losses saved to losses.pkl")
  else:
    optimize(toi=True, visualize=True)
  
  
  clear_states()
  forward('final')


if __name__ == '__main__':
  main()
