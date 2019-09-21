import taichi_lang as ti
import random
import sys
import math
import numpy as np
import cv2
import os
import taichi as tc
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)

max_steps = 512
vis_interval = 8
output_vis_interval = 8
steps = 512

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

hidden = scalar()

x = vec()
v = vec()

n_gravitation = 4
goal = vec()
gravitation = scalar()

n_hidden = 64

weight1 = scalar()
bias1 = scalar()
weight2 = scalar()
bias2 = scalar()

pad = 0.1
gravitation_position = [[pad, pad], [pad, 1-pad], [1-pad, 1-pad], [1-pad, pad]]


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).place(x, v)
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_hidden).place(hidden)
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_gravitation).place(gravitation)
  ti.root.dense(ti.ij, (6, n_hidden)).place(weight1)
  ti.root.dense(ti.i, n_hidden).place(bias1)
  ti.root.dense(ti.ij, (n_hidden, n_gravitation)).place(weight2)
  ti.root.dense(ti.i, n_gravitation).place(bias2)
  ti.root.place(loss)
  ti.root.place(goal)
  ti.root.lazy_grad()


dt = 0.03
alpha = 0.00000
learning_rate = 1e-3

K = 1e-3


@ti.kernel
def nn1(t: ti.i32):
  for i in range(n_hidden):
    act = 0.0
    act += (x[t][0] - 0.5) * weight1[0, i]
    act += (x[t][1] - 0.5) * weight1[1, i]
    act += v[t][0] * weight1[2, i]
    act += v[t][1] * weight1[3, i]
    act += (goal[None][0] - 0.5) * weight1[4, i]
    act += (goal[None][1] - 0.5) * weight1[5, i]
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
    v[t] = v[t - 1] + dt * gravitational_force
    x[t] = x[t - 1] + dt * v[t]


@ti.kernel
def compute_loss(t: ti.i32):
  loss[None] = (x[t] - goal[None]).norm_sqr()


gui = tc.core.GUI("Gravity", tc.Vectori(1024, 1024))


def forward(visualize=False, output=None):
  interval = vis_interval
  if output:
    interval = output_vis_interval
    os.makedirs('gravity/{}/'.format(output), exist_ok=True)
  
  canvas = gui.get_canvas()
  for t in range(1, steps):
    nn1(t)
    nn2(t)
    advance(t)
    
    if (t + 1) % interval == 0 and visualize:
      canvas.clear(0x3C733F)
      
      for i in range(n_gravitation):
        r = (gravitation[t, i] + 1) * 30
        canvas.circle(tc.Vector(*gravitation_position[i])).radius(r).color(
          0xccaa44).finish()
      
      canvas.circle(tc.Vector(goal[None][0], goal[None][1])).radius(10).color(
        0x3344cc).finish()
      canvas.circle(tc.Vector(x[t][0], x[t][1])).radius(10).color(
        0xF20530).finish()
      
      gui.update()
      if output:
        gui.screenshot('gravity/{}/{:04d}.png'.format(output, t))
  
  compute_loss(steps - 1)




def rand():
  return 0.2 + random.random() * 0.6
tasks = [((rand(), rand()), (rand(), rand())) for i in range(10)]

def initialize():
  x[0] = [rand(), rand()]
  goal[None] = [rand(), rand()]
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
    vis = iter % 100 == 0
    with ti.Tape(loss):
      forward(visualize=vis)
    losses.append(loss[None])
    # print(iter, "loss", loss[None])
    if vis:
      print(iter, sum(losses))
      losses.clear()

    for i in range(6):
      for j in range(n_hidden):
        weight1[i, j] = weight1[i, j] - weight1.grad[i, j] * learning_rate
    for j in range(n_hidden):
      bias1[j] = bias1[j] - bias1.grad[j] * learning_rate
      
    for i in range(n_hidden):
      for j in range(n_gravitation):
        weight2[i, j] = weight2[i, j] - weight2.grad[i, j] * learning_rate
    for j in range(n_gravitation):
      bias2[j] = bias2[j] - bias2.grad[j] * learning_rate
  
  forward(visualize=True, output='final')


if __name__ == '__main__':
  for i in range(6):
    for j in range(n_hidden):
      weight1[i, j] = (random.random() - 0.5) * 0.1
  for i in range(n_hidden):
    for j in range(n_gravitation):
      weight2[i, j] = (random.random() - 0.5) * 0.1
  optimize()
