import taichi_lang as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)

max_steps = 2048
vis_interval = 2
output_vis_interval = 2
steps = 1024
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

ti.cfg.arch = ti.cuda

x = vec()
v = vec()
rotation = scalar()
# angular velocity
omega = scalar()
halfsize = vec()

inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
omega_inc = scalar()

n_objects = 1
# target_ball = 0
elasticity = 0.8
ground_height = 0.1
gravity = -9.8


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, rotation,
                                                              omega, v_inc,
                                                              omega_inc)
  ti.root.dense(ti.i, n_objects).place(halfsize, inverse_mass, inverse_inertia)
  ti.root.place(loss)
  ti.root.lazy_grad()


dt = 0.001
learning_rate = 0.02


@ti.func
def rotation_matrix(r):
  return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.func
def initialize_properties():
  for i in range(n_objects):
    inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
    inverse_inertia[i] = 1.0 / (4 / 3 * halfsize[i][0] * halfsize[i][1] * (
        halfsize[i][0] * halfsize[i][0] + halfsize[i][1] * halfsize[i][1]))


@ti.kernel
def apply_gravity_and_collide(t: ti.i32):
  for i in range(n_objects):
    for k in ti.static(range(4)):
      # the corner for collision detection
      hs = halfsize[k]
      offset_scale = ti.Vector([k % 2 * 2 - 1, (k + 1) % 2 * 2 - 1])
      rot = rotation[i]
      rot_matrix = rotation_matrix(rot)
      
      rela_pos = offset_scale * hs @ rot_matrix
      rela_v = ti.Vector([-rela_pos[1], rela_pos[0]])
      
      corner_x = x[t, i] + rela_pos
      corner_v = v[t, i] + rela_v
      
      # Apply impulse so that there's no sinking
      normal = ti.Vector([0.0, 1.0])
      
      rela_v_ground = normal.dot(corner_v)
      
      impulse = 0
      if rela_v_ground < 0 and corner_x[1] < ground_height:
        impulse = 0
      # ti.atomic_add()


@ti.kernel
def advance(t: ti.i32):
  for i in range(n_objects):
    v[t, i] = v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector([0.0, 1.0])
    x[t, i] = x[t - 1, i] + dt * v[t, i]
    omega[t, i] = omega[t - 1, i] + omega_inc[t, i]
    rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
  pass


def forward(output=None):
  # initialize()
  
  for i in range(n_objects):
    x[0, i] = [0.5, 0.5]
    halfsize[i] = [0.1, 0.05]
    rotation[0, i] = 0.0
    omega[0, i] = 1
  
  initialize_properties()
  
  interval = vis_interval
  if output:
    interval = output_vis_interval
    os.makedirs('billiards/{}/'.format(output), exist_ok=True)
  
  for t in range(1, steps):
    # collide(t - 1)
    advance(t)
    
    if (t + 1) % interval == 0:
      img = np.ones(shape=(vis_resolution, vis_resolution, 3),
                    dtype=np.float32) * 0.8

      color = (0.3, 0.5, 0.8)
      for i in range(n_objects):
        points = []
        for k in range(4):
          offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
          rot = rotation[t, i]
          rot_matrix = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
          
          pos = np.array([x[t, i][0], x[t, i][1]]) + offset_scale * rot_matrix @ np.array(
            [halfsize[i][0], halfsize[i][1]])
          points.append(
            (int(pos[0] * vis_resolution), vis_resolution - int(pos[1] * vis_resolution)))
          
        print(points)
        
        cv2.fillConvexPoly(img, points=np.array(points), color=color)
      
      cv2.imshow('img', img)
      cv2.waitKey(1)
      if output:
        cv2.imwrite('rigid_body/{}/{:04d}.png'.format(output, t), img * 255)
  
  loss[None] = 0
  compute_loss(steps - 1)


@ti.kernel
def clear():
  for t in range(0, max_steps):
    for i in range(0, n_objects):
      x.grad[t, i] = ti.Vector([0.0, 0.0])
      v.grad[t, i] = ti.Vector([0.0, 0.0])
      rotation.grad[t, i] = 0.0
      omega.grad[t, i] = 0.0
      
      v_inc[t, i] = ti.Vector([0.0, 0.0])
      omega_inc[t, i] = 0.0
      
      v_inc.grad[t, i] = ti.Vector([0.0, 0.0])
      omega_inc.grad[t, i] = 0.0


def main():
  for iter in range(200):
    clear()
    tape = ti.tape()
    
    with tape:
      forward()
    
    print('Iter=', iter, 'Loss=', loss[None])
    
    # init_x.grad[None] = [0, 0]
    # init_v.grad[None] = [0, 0]
    # loss.grad[None] = -1
    
    # tape.grad()
  
  clear()
  # forward('final')


if __name__ == '__main__':
  main()
