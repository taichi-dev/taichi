import sys

import taichi_lang as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import cm
import taichi as tc

real = ti.f32
ti.set_default_fp(real)

max_steps = 4096
vis_interval = 16
output_vis_interval = 16
steps = 2048
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

x = vec()
v = vec()

goal = [0.9, 0.15]

n_objects = 1
ground_height = 0.1

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v)
  ti.root.place(loss)
  ti.root.lazy_grad()

dt = 0.0002
learning_rate = 1.0

@ti.kernel
def advance(t: ti.i32):
  for i in range(n_objects):
    new_v = ti.Vector([0.0, 0.0])
    if x[t, i][1] < ground_height:
      new_v = -v[t - 1, i]
    else:
      new_v = v[t - 1, i]
    v[t, i] = new_v
    x[t, i] = x[t - 1, i] + dt * new_v


@ti.kernel
def compute_loss(t: ti.i32):
  loss[None] = (x[t, 0] - ti.Vector(goal)).norm()


gui = tc.core.GUI("Rigid Body", tc.Vectori(1024, 1024))
canvas = gui.get_canvas()

def forward(output=None, visualize=True):
  interval = vis_interval
  total_steps = steps
  if output:
    interval = output_vis_interval
    # os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
    total_steps *= 2

  for t in range(1, total_steps):
    advance(t)
    
    if (t + 1) % interval == 0 and visualize:
      canvas.clear(0xFFFFFF)
      canvas.circle(tc.Vector(x[t, 0][0], x[t, 0][1])).radius(10).color(0x0).finish()
      offset = 0.003
      canvas.path(tc.Vector(0.05, ground_height - offset), tc.Vector(0.95, ground_height - offset)).radius(2).color(0xAAAAAA).finish()

      if output:
        gui.screenshot('rigid_body/{}/{:04d}.png'.format(output, t))

      gui.update()

  loss[None] = 0
  compute_loss(steps - 1)


def main():
  losses = []
  grads = []
  for i in range(0, 1):
    x[0, 0] = [0.7, 0.5]
    v[0, 0] = [-1, -2]
    
    with ti.Tape(loss):
      forward(visualize=True)
      
    print('Iter=', i, 'Loss=', loss[None])
    losses.append(loss[None])
  plt.plot(losses)
  plt.plot(grads)
  plt.show()
  
if __name__ == '__main__':
  main()
