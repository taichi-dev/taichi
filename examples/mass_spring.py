from mass_spring_robot_config import robots
import sys

import taichi_lang as ti
import math
import numpy as np
import cv2
import os

real = ti.f32
ti.set_default_fp(real)

max_steps = 4096
vis_interval = 256
output_vis_interval = 8
steps = 1024
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

x = vec()
v = vec()
v_inc = vec()

head_id = 0
goal = [0.9, 0.2]

n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -19.8
friction = 0.7
penalty = 1e4

gradient_clip = 1
spring_omega = 20
damping = 10
amplitude = 0.10

n_springs = 0
spring_anchor_a = ti.global_var(ti.i32)
spring_anchor_b = ti.global_var(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()

n_sin_waves = 10
weights = scalar()
bias = scalar()


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, v_inc)
  ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                       spring_length, spring_stiffness)
  ti.root.dense(ti.ij, (n_springs, n_sin_waves)).place(weights)
  ti.root.dense(ti.i, n_springs).place(bias)
  ti.root.place(loss)
  ti.root.lazy_grad()


dt = 0.001
learning_rate = 0.05

@ti.kernel
def apply_spring_force(t: ti.i32):
  for i in range(n_springs):
    a = spring_anchor_a[i]
    b = spring_anchor_b[i]
    pos_a = x[t, a]
    pos_b = x[t, b]
    dist = pos_a - pos_b
    length = dist.norm() + 1e-4
    
    actuation = 0.0
    for j in ti.static(range(n_sin_waves)):
      actuation += weights[i, j] * ti.sin(
        spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)
    actuation += bias[i]
    actuation = ti.tanh(actuation)
    
    target_length = spring_length[i] * (1.0 + amplitude * actuation)
    impulse = dt * (length - target_length) * spring_stiffness[
      i] / length * dist
    
    ti.atomic_add(v_inc[t + 1, a],  -impulse)
    ti.atomic_add(v_inc[t + 1, b],  impulse)

@ti.kernel
def advance(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * damping)
    new_v = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector([0.0, 1.0])
    depth = x[t - 1, i][1] - ground_height
    if depth < 0 and new_v[1] < 0:
      # friction projection
      if new_v[0] > 0:
        new_v[0] -= min(new_v[0], friction * -new_v[1])
      if new_v[0] < 0:
        new_v[0] += min(-new_v[0], friction * -new_v[1])
      new_v[1] = 0
    v[t, i] = new_v
    x[t, i] = x[t - 1, i] + dt * v[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
  loss[None] = (x[t, head_id] - ti.Vector(goal)).norm()


def forward(output=None):
  interval = vis_interval
  if output:
    interval = output_vis_interval
    os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)
  
  for t in range(1, steps):
    apply_spring_force(t - 1)
    advance(t)
    
    if (t + 1) % interval == 0:
      img = np.ones(shape=(vis_resolution, vis_resolution, 3),
                    dtype=np.float32) * 0.8
      
      y = int((1 - ground_height) * vis_resolution)
      cv2.line(img, (0, y), (vis_resolution - 2, y), color=(0.1, 0.1, 0.1),
               thickness=4)
      
      def circle(x, y, color):
        radius = 0.02
        cv2.circle(img, center=(
          int(vis_resolution * x), int(vis_resolution * (1 - y))),
                   radius=int(radius * vis_resolution), color=color,
                   thickness=-1)
      
      for i in range(n_objects):
        color = (0.4, 0.6, 0.6)
        if i == head_id:
          color = (0.8, 0.2, 0.3)
        circle(x[t, i][0], x[t, i][1], color)
      circle(goal[0], goal[1], (0.6, 0.2, 0.2))
      
      for i in range(n_springs):
        def get_pt(x):
          return int(x[0] * vis_resolution), int(vis_resolution - x[1]* vis_resolution)
        act = 0
        cv2.line(img, get_pt(x[t, spring_anchor_a[i]]), get_pt(x[t, spring_anchor_b[i]]), (0.5 + act, 0.5, 0.5 - act), thickness=6)
      
      cv2.imshow('img', img)
      cv2.waitKey(1)
      if output:
        cv2.imwrite('mass_spring/{}/{:04d}.png'.format(output, t), img * 255)
  
  loss[None] = 0
  compute_loss(steps - 1)


@ti.kernel
def clear_states():
  for t in range(0, max_steps):
    for i in range(0, n_objects):
      x.grad[t, i] = ti.Vector([0.0, 0.0])
      v.grad[t, i] = ti.Vector([0.0, 0.0])
      v_inc[t, i] = ti.Vector([0.0, 0.0])
      v_inc.grad[t, i] = ti.Vector([0.0, 0.0])


@ti.kernel
def clear_weights():
  for i in range(n_springs):
    bias.grad[i] = 0.0
    for j in range(n_sin_waves):
      weights.grad[i, j] = 0.0


def clear():
  clear_states()
  clear_weights()


def setup_robot(objects, springs):
  global n_objects, n_springs
  n_objects = len(objects)
  n_springs = len(springs)
  
  print('n_objects=', n_objects, '   n_springs=', n_springs)
  
  for i in range(n_objects):
    x[0, i] = objects[i]
  
  for i in range(n_springs):
    s = springs[i]
    spring_anchor_a[i] = s[0]
    spring_anchor_b[i] = s[1]
    spring_length[i] = s[2]
    spring_stiffness[i] = s[3]

def main():
  robot_id = 0
  if len(sys.argv) != 2:
    print("Usage: python3 mass_spring.py [robot_id=0, 1, 2, ...]")
  else:
    robot_id = int(sys.argv[1])
  setup_robot(*robots[robot_id]())
  
  for i in range(n_springs):
    for j in range(n_sin_waves):
      weights[i, j] = np.random.randn() * 0.001
  
  forward('initial')
  for iter in range(600):
    clear()
    loss.grad[None] = -1
    
    tape = ti.tape()
    
    with tape:
      forward()
    
    tape.grad()
    
    print('Iter=', iter, 'Loss=', loss[None])
    
    total_norm_sqr = 0
    for i in range(n_springs):
      for j in range(n_sin_waves):
        total_norm_sqr += weights.grad[i, j] ** 2
      total_norm_sqr += bias.grad[i] ** 2
    
    print(total_norm_sqr)
    
    scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
    for i in range(n_springs):
      for j in range(n_sin_waves):
        weights[i, j] += scale * weights.grad[i, j]
      bias[i] += scale * bias.grad[i]
  
  clear()
  forward('final')


if __name__ == '__main__':
  main()