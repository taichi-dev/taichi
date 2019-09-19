import taichi_lang as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)
# ti.runtime.print_preprocessed = True

n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
max_steps = 128
vis_interval = 32
output_vis_interval = 2
steps = 64
assert steps * 2 <= max_steps
amplify = 2

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

p = scalar()
rendered = scalar()
target = scalar()
initial = scalar()
loss = scalar()


# ti.cfg.arch = ti.cuda

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.ij, n_grid).place(p)
  ti.root.dense(ti.ij, n_grid).place(rendered)
  ti.root.dense(ti.ij, n_grid).place(target)
  ti.root.dense(ti.ij, n_grid).place(initial)
  ti.root.place(loss)
  ti.root.lazy_grad()


c = 340
# damping
alpha = 0.00000
inv_dx2 = inv_dx * inv_dx
dt = (math.sqrt(alpha * alpha + dx * dx / 3) - alpha) / c
learning_rate = 0.1


# TODO: there may by out-of-bound accesses here
@ti.func
def laplacian(t, i, j):
  return inv_dx2 * (
      -4 * p[t, i, j] + p[t, i, j - 1] + p[t, i, j + 1] + p[t, i + 1, j] +
      p[t, i - 1, j])


@ti.func
def gradient(t, i, j):
  return inv_dx * ti.Vector(
    [p[t, i, j + 1] - p[t, i, j - 1], p[t, i, j + 1] + p[t, i, j - 1]])


@ti.kernel
def initialize():
  for i in range(n_grid):
    for j in range(n_grid):
      p[0, i, j] = initial[i, j]


@ti.kernel
def fdtd(t: ti.i32):
  for i in range(n_grid):  # Parallelized over GPU threads
    for j in range(n_grid):
      laplacian_p = laplacian(t - 2, i, j)
      laplacian_q = laplacian(t - 1, i, j)
      p[t, i, j] = 2 * p[t - 1, i, j] + (
          c * c * dt * dt + c * alpha * dt) * laplacian_q - p[
                     t - 2, i, j] - c * alpha * dt * laplacian_p


@ti.kernel
def render_reflect(t: ti.i32):
  for i in range(n_grid):  # Parallelized over GPU threads
    for j in range(n_grid):
      grad = gradient(t, i, j)
      normal = ti.Vector.normalized(ti.Vector([grad[0], 1.0, grad[1]]))
      rendered[i, j] = normal[1]


@ti.func
def pattern(i, j):
  return ti.cast(ti.floor(i / (n_grid / 8)) + ti.floor(j / (n_grid / 8)),
                 ti.i32) % 2


@ti.kernel
def render_refract(t: ti.i32):
  for i in range(n_grid):  # Parallelized over GPU threads
    for j in range(n_grid):
      grad = gradient(t, i, j)

      scale = 2.0
      sample_x = i - grad[0] * scale
      sample_y = j - grad[1] * scale
      sample_x = ti.min(n_grid - 1, ti.max(0, sample_x))
      sample_y = ti.min(n_grid - 1, ti.max(0, sample_y))
      sample_xi = ti.cast(ti.floor(sample_x), ti.i32)
      sample_yi = ti.cast(ti.floor(sample_y), ti.i32)

      frac_x = sample_x - sample_xi
      frac_y = sample_y - sample_yi

      rendered[i, j] = (1.0 - frac_x) * (
          (1 - frac_y) * target[sample_xi, sample_yi] + frac_y * target[
                         sample_xi, sample_yi + 1]) + frac_x * (
        (1 - frac_y) * target[sample_xi + 1, sample_yi] + frac_y * target[
                         sample_xi + 1, sample_yi + 1]
      )

@ti.kernel
def clear_photon_map():
  for i in range(n_grid):
    for j in range(n_grid):
      rendered[i, j] = 0.0

@ti.kernel
def render_photon_map(t: ti.i32):
  for i in range(n_grid):  # Parallelized over GPU threads
    for j in range(n_grid):
      grad = gradient(t, i, j)

      scale = 5.0
      sample_x = i - grad[0] * scale
      sample_y = j - grad[1] * scale
      sample_x = ti.min(n_grid - 1, ti.max(0, sample_x))
      sample_y = ti.min(n_grid - 1, ti.max(0, sample_y))
      sample_xi = ti.cast(ti.floor(sample_x), ti.i32)
      sample_yi = ti.cast(ti.floor(sample_y), ti.i32)

      frac_x = sample_x - sample_xi
      frac_y = sample_y - sample_yi

      x = sample_xi
      y = sample_yi

      ti.atomic_add(rendered[x, y], (1 - frac_x) * (1 - frac_y))
      ti.atomic_add(rendered[x, y + 1], (1 - frac_x) * frac_y)
      ti.atomic_add(rendered[x + 1, y], frac_x * (1 - frac_y))
      ti.atomic_add(rendered[x + 1, y + 1], frac_x * frac_y)


@ ti.kernel
def compute_loss(t: ti.i32):
  for i in range(n_grid):
    for j in range(n_grid):
      ti.atomic_add(loss, dx * dx * ti.sqr(target[i, j] - p[t, i, j]))


@ti.kernel
def apply_grad():
  # gradient descent
  for i, j in initial.grad:
    initial[i, j] -= learning_rate * initial.grad[i, j]


def forward(output=None):
  steps_mul = 1
  interval = vis_interval
  if output:
    os.makedirs(output, exist_ok=True)
    steps_mul = 2
    interval = output_vis_interval
  initialize()
  for t in range(2, steps * steps_mul):
    fdtd(t)
    if (t + 1) % interval == 0:
      img = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
      clear_photon_map()
      render_photon_map(t)
      for i in range(n_grid):
        for j in range(n_grid):
          img[i, j] = rendered[i, j] * 0.3
      img = cv2.resize(img, fx=4, fy=4, dsize=None)
      cv2.imshow('img', img)
      cv2.waitKey(1)
      if output:
        img = np.clip(img, 0, 255)
        cv2.imwrite(output + "/{:04d}.png".format(t), img * 255)
  loss[None] = 0
  compute_loss(steps - 1)


def main():
  # initialization
  target_img = cv2.imread('iclr2020.png')[:, :, 0] / 255.0
  target_img = cv2.resize(target_img, (n_grid, n_grid))
  target_img -= target_img.mean()
  cv2.imshow('target', target_img * amplify + 0.5)
  # print(target_img.min(), target_img.max())
  for i in range(n_grid):
    for j in range(n_grid):
      target[i, j] = float(target_img[i, j])

  # initial[n_grid // 2, n_grid // 2] = 1
  forward('initial')

  for opt in range(200):
    with ti.Tape(loss):
      forward()

    print('Iter', opt, ' Loss =', loss[None])

    apply_grad()

  forward('optimized')


if __name__ == '__main__':
  main()
