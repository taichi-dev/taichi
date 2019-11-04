import taichi as ti
import random
import sys
import math
import numpy as np
import os
import taichi as tc
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)

max_steps = 2048
vis_interval = 8
output_vis_interval = 8
steps = 512
seg_size = 256

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

hidden = scalar()
damping = 0.2

x = vec()
v = vec()

n_gravitation = 8
goal = vec()
goal_v = vec()
gravitation = scalar()

n_hidden = 64

weight1 = scalar()
bias1 = scalar()
weight2 = scalar()
bias2 = scalar()

pad = 0.1
gravitation_position = [[pad, pad], [pad, 1 - pad], [1 - pad, 1 - pad],
                        [1 - pad, pad], [0.5, 1-pad], [0.5, pad], [pad, 0.5], [1-pad, 0.5]]


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).place(x, v)
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_hidden).place(hidden)
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_gravitation).place(gravitation)
  ti.root.dense(ti.ij, (8, n_hidden)).place(weight1)
  ti.root.dense(ti.i, n_hidden).place(bias1)
  ti.root.dense(ti.ij, (n_hidden, n_gravitation)).place(weight2)
  ti.root.dense(ti.i, n_gravitation).place(bias2)
  ti.root.place(loss)
  ti.root.dense(ti.i, max_steps).place(goal, goal_v)
  ti.root.lazy_grad()


dt = 0.03
alpha = 0.00000
learning_rate = 2e-2

K = 1e-3


@ti.kernel
def nn1(t: ti.i32):
  for i in range(n_hidden):
    act = 0.0
    act += (x[t][0] - 0.5) * weight1[0, i]
    act += (x[t][1] - 0.5) * weight1[1, i]
    act += v[t][0] * weight1[2, i]
    act += v[t][1] * weight1[3, i]
    act += (goal[t][0] - 0.5) * weight1[4, i]
    act += (goal[t][1] - 0.5) * weight1[5, i]
    act += (goal_v[t][0] - 0.5) * weight1[6, i]
    act += (goal_v[t][1] - 0.5) * weight1[7, i]
    act += bias1[i]
    hidden[t, i] = ti.tanh(act)


@ti.kernel
def nn2(t: ti.i32):
  for i in range(n_gravitation):
    act = 0.0
    for j in ti.static(range(n_hidden)):
      act += hidden[t, j] * weight2[j, i]
    act += bias2[i]
    gravitation[t, i] = ti.tanh(act)


@ti.kernel
def advance(t: ti.i32):
  for _ in range(1):  # parallelize this loop
    gravitational_force = ti.Vector([0.0, 0.0])
    for i in ti.static(range(n_gravitation)):  # instead of this one
      r = x[t - 1] - ti.Vector(gravitation_position[i])
      len_r = ti.max(r.norm(), 1e-1)
      gravitational_force += K * gravitation[t, i] / (len_r * len_r * len_r) * r
    v[t] = v[t - 1] * math.exp(-dt * damping) + dt * gravitational_force
    x[t] = x[t - 1] + dt * v[t]


@ti.kernel
def compute_loss(t: ti.i32):
  ti.atomic_add(loss[None], dt * (x[t] - goal[t]).norm_sqr())


gui = tc.core.GUI("Electric", tc.veci(1024, 1024))


def forward(visualize=False, output=None):
  interval = vis_interval
  if output:
    interval = output_vis_interval
    os.makedirs('electric/{}/'.format(output), exist_ok=True)
  
  canvas = gui.get_canvas()
  for t in range(1, steps):
    nn1(t)
    nn2(t)
    advance(t)
    compute_loss(t)
    
    if (t + 1) % interval == 0 and visualize:
      canvas.clear(0x3C733F)
      
      for i in range(n_gravitation):
        r = (gravitation[t, i] + 1) * 30
        canvas.circle(tc.vec(*gravitation_position[i])).radius(r).color(
          0xccaa44).finish()
      
      canvas.circle(tc.vec(x[t][0], x[t][1])).radius(30).color(
        0xF20530).finish()
      
      canvas.circle(tc.vec(goal[t][0], goal[t][1])).radius(10).color(
        0x3344cc).finish()
      
      gui.update()
      if output:
        gui.screenshot('electric/{}/{:04d}.png'.format(output, t))


def rand():
  return 0.2 + random.random() * 0.6


tasks = [((rand(), rand()), (rand(), rand())) for i in range(10)]

def lerp(x, a, b):
  return (1 - x) * a + x * b

def initialize():
  # x[0] = [rand(), rand()]
  segments = steps // seg_size
  points = []
  for i in range(segments + 1):
    points.append([rand(), rand()])
  for i in range(segments):
    for j in range(steps // segments):
      k = steps // segments * i + j
      goal[k] = [lerp(j / seg_size, points[i][0], points[i + 1][0]),
                 lerp(j / seg_size, points[i][1], points[i + 1][1])]
      goal_v[k] = [points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]]
  x[0] = points[0]
  # x[0] = [0.3, 0.6]
  # goal[None] = [0.5, 0.2]
  # i = random.randrange(2)
  # x[0] = tasks[i][0]
  # goal[None] = tasks[i][1]


def optimize():
  initialize()
  forward(visualize=True, output='initial')
  
  losses = []
  for iter in range(200000):
    initialize()
    vis = iter % 200 == 0
    output = None
    if vis:
      output = 'iter{:05d}'.format(iter)
    with ti.Tape(loss):
      forward(visualize=vis, output=output)
    losses.append(loss[None])
    # print(iter, "loss", loss[None])
    if vis:
      print(iter, sum(losses))
      losses.clear()
    
    tot = 0
    for i in range(8):
      for j in range(n_hidden):
        weight1[i, j] = weight1[i, j] - weight1.grad[i, j] * learning_rate
        tot += weight1.grad[i, j] ** 2
    # print(tot)
    for j in range(n_hidden):
      bias1[j] = bias1[j] - bias1.grad[j] * learning_rate
    
    for i in range(n_hidden):
      for j in range(n_gravitation):
        weight2[i, j] = weight2[i, j] - weight2.grad[i, j] * learning_rate
    for j in range(n_gravitation):
      bias2[j] = bias2[j] - bias2.grad[j] * learning_rate
  
  forward(visualize=True, output='final')


if __name__ == '__main__':
  for i in range(8):
    for j in range(n_hidden):
      weight1[i, j] = (random.random() - 0.5) * 0.3
  for i in range(n_hidden):
    for j in range(n_gravitation):
      weight2[i, j] = (random.random() - 0.5) * 0.3
  optimize()
