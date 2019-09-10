import taichi_lang as ti
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)
# ti.runtime.print_preprocessed = True

n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
max_steps = 512
vis_interval = 16
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
  ti.root.dense(ti.ij, n_grid).place(target.grad)
  ti.root.dense(ti.ij, n_grid).place(initial)
  ti.root.dense(ti.ij, n_grid).place(initial.grad)
  ti.root.place(loss)
  ti.root.place(loss.grad)
  # ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
  # ti.root.place(init_v, loss, x_avg)
  # ti.root.lazy_grad()


c = 340
# damping
alpha = 0.0000
inv_dx2 = inv_dx * inv_dx
dt = (math.sqrt(alpha * alpha + dx * dx / 3) - alpha) / c
learning_rate = 0.1


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
  initialize()
  for t in range(2, max_steps):
    ftdt(t)
    if (t + 1) % vis_interval == 0:
      img = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
      for i in range(n_grid):
        for j in range(n_grid):
          img[i, j] = (p[t, i, j] - 0.5) * amplify + 0.5
      img = cv2.resize(img, fx=4, fy=4, dsize=None)
      cv2.imshow('img', img)
      cv2.waitKey(1)

@ti.kernel
def apply_grad():
  # gradient descent
  for i, j in initial.grad:
    initial[i, j] -= learning_rate * initial.grad[i, j]

def backward():
  clear_p_grad()
  clear_initial_grad()
  for t in reversed(range(2, max_steps)):
    ftdt.grad(t)
  initialize.grad()
  apply_grad()

@ti.kernel
def clear_p_grad():
  for t, i, j in p:
    p.grad[t, i, j] = 0

@ti.kernel
def clear_initial_grad():
  for i, j in initial:
    initial.grad[i, j] = 0

def main():
  # initialization
  target_img = cv2.imread('iclr2020.png')[:,:,0] / 255.0
  # print(target_img.min(), target_img.max())
  for i in range(n_grid):
    for j in range(n_grid):
      target[i, j] = float(target_img[i, j])

  # initial[n_grid // 2, n_grid // 2] = 1

  forward()
  print('Loss =', loss[None])
  loss.grad[None] = 1
  backward()

if __name__ == '__main__':
  main()
