from mass_spring_robot_config import robots
import sys

import matplotlib.pyplot as plt
import taichi_lang as ti
import math
import numpy as np
import cv2
import os

real = ti.f32
ti.set_default_fp(real)
# ti.cfg.print_ir = True

max_steps = 4096
vis_interval = 256
output_vis_interval = 8
steps = 2048 // 3
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
gravity = -4.8
friction = 2.5

gradient_clip = 1
spring_omega = 10
damping = 15

n_springs = 0
spring_anchor_a = ti.global_var(ti.i32)
spring_anchor_b = ti.global_var(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 32
weights2 = scalar()
bias2 = scalar()
hidden = scalar()

act = scalar()

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, v_inc)
  ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                       spring_length, spring_stiffness,
                                       spring_actuation)
  ti.root.dense(ti.ij, (n_springs, n_sin_waves)).place(weights1)
  ti.root.dense(ti.i, n_springs).place(bias1)
  ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
  ti.root.dense(ti.i, n_springs).place(bias2)
  ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
  ti.root.dense(ti.ij, (max_steps, n_springs)).place(act)
  ti.root.place(loss)
  ti.root.lazy_grad()


dt = 0.004
learning_rate = 25

@ti.kernel
def nn1(t: ti.i32):
  for i in range(n_hidden):
    actuation = 0.0
    for j in ti.static(range(n_sin_waves)):
      actuation += weights1[i, j] * ti.sin(
        spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)
    actuation += bias1[i]
    actuation = ti.tanh(actuation)
    hidden[t, i] = actuation
    
@ti.kernel
def nn2(t: ti.i32):
  for i in range(n_springs):
    actuation = 0.0
    for j in ti.static(range(n_hidden)):
      actuation += weights2[i, j] * hidden[t, j]
    actuation += bias2[i]
    actuation = ti.tanh(actuation)
    act[t, i] = actuation

@ti.kernel
def apply_spring_force(t: ti.i32):
  for i in range(n_springs):
    a = spring_anchor_a[i]
    b = spring_anchor_b[i]
    pos_a = x[t, a]
    pos_b = x[t, b]
    dist = pos_a - pos_b
    length = dist.norm() + 1e-4
    
    target_length = spring_length[i] * (1.0 + spring_actuation[i] * act[t, i])
    impulse = dt * (length - target_length) * spring_stiffness[
      i] / length * dist
    
    ti.atomic_add(v_inc[t + 1, a], -impulse)
    ti.atomic_add(v_inc[t + 1, b], impulse)

use_toi = False

@ti.kernel
def advance_toi(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * damping)
    old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0]) + v_inc[t, i]
    old_x = x[t - 1, i]
    new_x = old_x + dt * old_v
    toi = 0.0
    new_v = old_v
    if new_x[1] < ground_height and old_v[1] < 1e-4:
      toi = -(old_x[1] - ground_height) / old_v[1]
      new_v = ti.Vector([0.0, 0.0])
    new_x = old_x + toi * old_v + (dt - toi) * new_v
    
    v[t, i] = new_v
    x[t, i] = new_x
  
@ti.kernel
def advance_no_toi(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * damping)
    old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0]) + v_inc[t, i]
    old_x = x[t - 1, i]
    new_v = old_v
    depth = old_x[1] - ground_height
    if depth < 0 and new_v[1] < 0:
      # friction projection
      new_v[0] = 0
      new_v[1] = 0
    new_x = old_x + dt * new_v
    v[t, i] = new_v
    x[t, i] = new_x


@ti.kernel
def compute_loss(t: ti.i32):
  loss[None] = -x[t, head_id][0]


def forward(output=None, visualize=True):
  interval = vis_interval
  if output:
    interval = output_vis_interval
    os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)
  
  total_steps = steps if not output else steps * 2
  
  for t in range(1, total_steps):
    nn1(t - 1)
    nn2(t - 1)
    apply_spring_force(t - 1)
    if use_toi:
      advance_toi(t)
    else:
      advance_no_toi(t)
    
    if (t + 1) % interval == 0 and visualize:
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
          return int(x[0] * vis_resolution), int(
            vis_resolution - x[1] * vis_resolution)
        
        act = 0
        cv2.line(img, get_pt(x[t, spring_anchor_a[i]]),
                 get_pt(x[t, spring_anchor_b[i]]), (0.5 + act, 0.5, 0.5 - act),
                 thickness=6)
      
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

def clear():
  clear_states()


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
    spring_actuation[i] = s[4]

def optimize(toi, visualize):
  global use_toi
  use_toi = toi
  for i in range(n_hidden):
    for j in range(n_sin_waves):
      weights1[i, j] = np.random.randn() * 0.1
      
  for i in range(n_springs):
    for j in range(n_hidden):
      weights2[i, j] = np.random.randn() * 0.1

  losses = []
  forward('initial', visualize=visualize)
  for iter in range(100):
    clear()
  
    with ti.Tape(loss):
      forward(visualize=visualize)
  
    print('Iter=', iter, 'Loss=', loss[None])
  
    total_norm_sqr = 0
    for i in range(n_springs):
      for j in range(n_sin_waves):
        total_norm_sqr += weights1.grad[i, j] ** 2
      total_norm_sqr += bias1.grad[i] ** 2
  
    print(total_norm_sqr)
  
    # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
    gradient_clip = 0.1
    scale = gradient_clip / (total_norm_sqr ** 0.5 + 1e-6)
    for i in range(n_hidden):
      for j in range(n_sin_waves):
        weights1[i, j] -= scale * weights1.grad[i, j]
      bias1[i] -= scale * bias1.grad[i]
      
    for i in range(n_springs):
      for j in range(n_hidden):
        weights2[i, j] -= scale * weights2.grad[i, j]
      bias2[i] -= scale * bias2.grad[i]
    losses.append(loss[None])

  return losses

def main():
  robot_id = 0
  if len(sys.argv) != 3:
    print("Usage: python3 mass_spring.py [robot_id=0, 1, 2, ...] [task]")
  else:
    robot_id = int(sys.argv[1])
    task = sys.argv[2]

  setup_robot(*robots[robot_id]())

  if task == 'plot':
    for toi in [False, True]:
      for i in range(5):
        losses = optimize(toi=toi, visualize=False)
        plt.plot(losses, 'g' if toi else 'r')

    plt.title('Mass Spring (Red is no TOI, green is TOI)')
    plt.xlabel("Iteration")
    plt.ylabel("loss")
    plt.show()
  else:
    optimize(toi=True, visualize=True)
    clear()
    forward('final')



if __name__ == '__main__':
  main()
