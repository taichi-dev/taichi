import taichi_lang as ti
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)

n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
max_steps = 1024
steps = max_steps
gravity = 9.8
amplify = 2

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

p = scalar()
target = scalar()
initial = scalar()
loss = scalar()

ti.cfg.arch = ti.cuda

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.ij, n_grid).place(p)
  ti.root.dense(ti.l, max_steps).dense(ti.ij, n_grid).place(p.grad)
  ti.root.dense(ti.ij, n_grid).place(target)
  ti.root.dense(ti.ij, n_grid).place(initial)
  ti.root.place(loss)
  # ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
  # ti.root.place(init_v, loss, x_avg)
  # ti.root.lazy_grad()


c = 340
# damping
alpha = 0.0000
inv_dx2 = inv_dx * inv_dx
dt = (math.sqrt(alpha * alpha + dx * dx / 3) - alpha) / c


@ti.func
def laplacian(t, i, j):
  return inv_dx2 * (
      -4 * p[t, i, j] + p[t, i, j - 1] + p[t, i, j + 1] + p[t, i + 1, j] +
      p[t, i - 1, j])

@ti.kernel
def initialize():
  for i in range(n_grid):
    for j in range(n_grid):
      p[0, i, j] = initial[i, j]


@ti.kernel
def ftdt(t: ti.i32):
  for i in range(n_grid):
    for j in range(n_grid):
      laplacian_p = laplacian(t - 2, i, j)
      laplacian_q = laplacian(t - 1, i, j)
      p[t, i, j] = 2 * p[t - 1, i, j] + (
          c * c * dt * dt + c * alpha * dt) * laplacian_q - p[
                     t - 2, i, j] - c * alpha * dt * laplacian_p

@ti.kernel
def compute_loss():
  for i in range(n_grid):
    for j in range(n_grid):
      ti.atomic_add(loss, ti.sqr(target[i, j] - p[steps - 1]))



def forward():
  for t in range(2, max_steps):
    ftdt(t)
    if t % 8 == 0:
      img = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
      for i in range(n_grid):
        for j in range(n_grid):
          img[i, j] = (p[t, i, j] - 0.5) * amplify + 0.5
      img = cv2.resize(img, fx=4, fy=4, dsize=None)
      cv2.imshow('img', img)
      cv2.waitKey(1)

def backward():
  for t in reversed(range(2, max_steps)):
    ftdt.grad(t)

def main():
  # initialization
  target_img = cv2.imread('iclr2020.png')[:,:,0] / 255.0
  for i in range(n_grid):
    for j in range(n_grid):
      target[i, j] = float(target_img[i, j])

  initial[n_grid // 2, n_grid // 2] = 1

  initialize()
  forward()
  backward()


if __name__ == '__main__':
  main()
