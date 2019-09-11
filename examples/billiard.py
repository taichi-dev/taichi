import taichi_lang as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)
# ti.runtime.print_preprocessed = True

dt = 3e-4
max_steps = 512
vis_interval = 32
output_vis_interval = 2
steps = 256
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

# initial = scalar()
# loss = scalar()

# ti.cfg.arch = ti.cuda

x = vec()
v = vec()
impulse = vec()

n_balls = 2
radius = 0.05
elasticity = 0.8


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_balls).place(x, v, impulse)
  ti.root.lazy_grad()


alpha = 0.00000
dt = 0.003
penalty = 0.1
learning_rate = 0.1


@ti.kernel
def collide(t: ti.i32):
  for i in range(n_balls):
    impulse_contribution = ti.Vector([0.0, 0.0])
    for j in range(n_balls):
      if i != j:
        dist = x[t, i] - x[t, j]
        dist_norm = dist.norm()
        if dist_norm < 2 * radius:
          dir = ti.Vector.normalized(dist)
          rela_v = v[t, i] - v[t, j]
          projected_v = dir.dot(rela_v)
          if projected_v < 0:
            imp = -(1 + elasticity) * 0.5 * projected_v
            impulse_contribution += imp * dir
            
          # impulse_contribution -= dir * (penalty * (dist_norm - 2 * radius))
    
    impulse[t + 1, i] = impulse_contribution


@ti.kernel
def advance(t: ti.i32):
  for i in range(n_balls):
    v[t, i] = v[t - 1, i] + impulse[t, i]
    x[t, i] = x[t - 1, i] + dt * v[t, i]


'''
@ti.kernel
def compute_loss(t: ti.i32):
  for i in range(n_grid):
    for j in range(n_grid):
      ti.atomic_add(loss, dx * dx * ti.sqr(target[i, j] - p[t, i, j]))

@ti.kernel
def apply_grad():
  # gradient descent
  for i, j in initial.grad:
    initial[i, j] -= learning_rate * initial.grad[i, j]
'''


def forward(output=None):
  x[0, 0] = [0.3, 0.5]
  x[0, 1] = [0.7, 0.51]
  v[0, 0] = [1, 0]
  
  for t in range(1, steps):
    collide(t - 1)
    advance(t)
    
    img = np.ones(shape=(vis_resolution, vis_resolution, 3), dtype=np.float32)
    for i in range(n_balls):
      cv2.circle(img, center=(
      int(vis_resolution * x[t, i][0]), int(vis_resolution * x[t, i][1])),
                 radius=int(radius * vis_resolution), color=(0, 0, 0), thickness=-1)
    cv2.imshow('img', img)
    cv2.waitKey(1)
  
  # loss[None] = 0
  # compute_loss(steps - 1)


def main():
  forward()


if __name__ == '__main__':
  main()
