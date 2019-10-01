import taichi as ti
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
max_steps = 256
vis_interval = 32
output_vis_interval = 2
steps = 256
amplify = 2

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

p = scalar()
rendered = scalar()
target = scalar()
initial = scalar()
loss = scalar()
height_gradient = vec()

bottom_image = scalar()
refracted_image = scalar()

# mode = 'reflect'
mode = 'refract'

assert mode in ['reflect', 'refract', 'photon']

# ti.cfg.arch = ti.cuda

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.ij, n_grid).place(p)
  ti.root.dense(ti.ij, n_grid).place(rendered)
  ti.root.dense(ti.ij, n_grid).place(target)
  ti.root.dense(ti.ij, n_grid).place(initial)
  ti.root.dense(ti.ij, n_grid).place(height_gradient)
  ti.root.dense(ti.ijk, (n_grid, n_grid, 3)).place(bottom_image)
  ti.root.dense(ti.ijk, (n_grid, n_grid, 3)).place(refracted_image)
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
  return 0.5 * inv_dx * ti.Vector(
    [p[t, i + 1, j] - p[t, i - 1, j], p[t, i, j + 1] - p[t, i, j - 1]])


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
      grad = height_gradient[i, j]
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
      grad = height_gradient[i, j]
      
      scale = 2.0
      sample_x = i - grad[0] * scale
      sample_y = j - grad[1] * scale
      sample_x = ti.min(n_grid - 1, ti.max(0, sample_x))
      sample_y = ti.min(n_grid - 1, ti.max(0, sample_y))
      sample_xi = ti.cast(ti.floor(sample_x), ti.i32)
      sample_yi = ti.cast(ti.floor(sample_y), ti.i32)
      
      frac_x = sample_x - sample_xi
      frac_y = sample_y - sample_yi
      
      for k in ti.static(range(3)):
        refracted_image[i, j, k] = (1.0 - frac_x) * (
            (1 - frac_y) * bottom_image[sample_xi, sample_yi, k] + frac_y *
            bottom_image[
              sample_xi, sample_yi + 1, k]) + frac_x * (
                                       (1 - frac_y) * bottom_image[
                                     sample_xi + 1, sample_yi, k] + frac_y *
                                       bottom_image[
                                         sample_xi + 1, sample_yi + 1, k]
                                   )


@ti.kernel
def compute_height_gradient(t: ti.i32):
  for i in range(n_grid):  # Parallelized over GPU threads
    for j in range(n_grid):
      # TODO: fix boundary
      height_gradient[i, j] = gradient(t, i, j)


@ti.kernel
def clear_photon_map():
  for i in range(n_grid):
    for j in range(n_grid):
      rendered[i, j] = 0.0


@ti.kernel
def render_photon_map(t: ti.i32, offset_x: ti.f32, offset_y: ti.f32):
  for i in range(n_grid):  # Parallelized over GPU threads
    for j in range(n_grid):
      grad = height_gradient[i, j] * (1 - offset_x) * (1 - offset_y) + \
             height_gradient[i + 1, j] * offset_x * (1 - offset_y) + \
             height_gradient[i, j + 1] * (1 - offset_x) * offset_y + \
             height_gradient[i + 1, j + 1] * offset_x * offset_y
      
      scale = 5.0
      sample_x = i - grad[0] * scale + offset_x
      sample_y = j - grad[1] * scale + offset_y
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


def forward(output=None):
  steps_mul = 1
  interval = vis_interval
  if output:
    os.makedirs(output, exist_ok=True)
    interval = output_vis_interval
  initialize()
  for t in range(2, steps * steps_mul):
    fdtd(t)
    if (t + 1) % interval == 0:
      clear_photon_map()
      compute_height_gradient()
      if mode == 'refract':
        render_refract()
      elif mode == 'photon':
        render_photon_map(t, 0.25, 0.25)
        render_photon_map(t, 0.25, 0.75)
        render_photon_map(t, 0.75, 0.25)
        render_photon_map(t, 0.75, 0.75)
      else:
        render_reflect()
      if mode == 'refract':
        img = np.zeros(shape=(n_grid, n_grid, 3), dtype=np.float32)
        for i in range(n_grid):
          for j in range(n_grid):
            for k in range(3):
              img[i, j, k] = refracted_image[i, j, k]
        img = cv2.resize(img, fx=4, fy=4, dsize=None)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        if output:
          img = np.clip(img, 0, 255)
          cv2.imwrite(output + "/{:04d}.png".format(t), img * 255)
      else:
        img = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
        for i in range(n_grid):
          for j in range(n_grid):
            img[i, j] = rendered[i, j] * 0.3 / 4
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
  bot_img = cv2.imread('squirrel.jpg') / 255.0
  for i in range(256):
    for j in range(256):
      for k in range(3):
        bottom_image[i, j, k] = bot_img[i, j, k]
  target_img = cv2.imread('iclr2020.png')[:, :, 0] / 255.0
  target_img = cv2.resize(target_img, (n_grid, n_grid))
  target_img -= target_img.mean()
  cv2.imshow('target', target_img * amplify + 0.5)
  # print(target_img.min(), target_img.max())
  for i in range(n_grid):
    for j in range(n_grid):
      target[i, j] = float(target_img[i, j])
  
  initial[n_grid // 2, n_grid // 2] = 1
  # forward('water_renderer/initial')
  initial[n_grid // 2, n_grid // 2] = 0
  
  from adversarial import vgg_grad, predict
  
  for opt in range(10):
    with ti.Tape(loss):
      forward()
      
      feed_to_vgg = np.zeros((224, 224, 3), dtype=np.float32)
      # Note: do a transpose here
      for i in range(224):
        for j in range(224):
          for k in range(3):
            feed_to_vgg[i, j, k] = refracted_image[i + 16, j + 16, 2-k]
      
      predict(feed_to_vgg)
      grad = vgg_grad(feed_to_vgg)
      for i in range(224):
        for j in range(224):
          for k in range(3):
            refracted_image.grad[i + 16, j + 16, k] = grad[i, j, 2-k] * 0.001
      
    
    print('Iter', opt, ' Loss =', loss[None])
    
    apply_grad()
  
  forward('water_renderer/optimized')


if __name__ == '__main__':
  main()
