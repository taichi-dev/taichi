import taichi_lang as ti
import numpy as np
import random
import cv2

real = ti.f32
dim = 2
n_particles = 4096
n_grid = 32
dx = 1.0 / n_grid
inv_dx = 1.0 / dx
dt = 1e-3

p_mass = 1.0
p_vol = 1.0


def scalar():
  return ti.var(dt=real)


def vec():
  return ti.Vector(dim, dt=real)


def mat():
  return ti.Matrix(dim, dim, dt=real)


x, v = vec(), vec()
grid_v = vec()
grid_m = scalar()


@ti.layout
def place():
  ti.root.dense(ti.k, n_particles).place(x, v)
  ti.root.dense(ti.ij, n_grid).place(grid_v, grid_m)
  ti.cfg().print_ir = True

@ti.kernel
def clear_grid():
  for i, j in grid_m:
    grid_v[i, j].assign(ti.Vector([0.0, 0.0]))
    grid_m[i, j].assign(0.0)


@ti.kernel
def p2g():
  for p in x(0):
    base_coord = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
    fx = x[p] * inv_dx - ti.cast(base_coord, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        # dpos = (ti.Vector([i, j]) - fx) * dx
        weight = w[i](0) * w[j](1)
        grid_v[base_coord(0) + i, base_coord(1) + j].assign(
          grid_v[base_coord(0) + i, base_coord(1) + j] + weight * p_mass * v[p])
        grid_m[base_coord(0) + i, base_coord(1) + j].assign(
          grid_m[base_coord(0) + i, base_coord(1) + j] + weight * p_mass)


@ti.kernel
def grid_op():
  for i, j in grid_m:
    if grid_m[i, j] > 0.0:
      inv_m = 1.0 / grid_m[i, j]
      grid_v[i, j] = inv_m * grid_v[i, j]


@ti.kernel
def g2p():
  for p in x(0):
    base_coord = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
    fx = x[p] * inv_dx - ti.cast(base_coord, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)]
    local_v = ti.Vector([0.0, 0.0])
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        weight = w[i](0) * w[j](1)
        local_v = local_v + weight * grid_v[base_coord(0) + i, base_coord(1) + j]
    v[p].assign(local_v)
    x[p] = x[p] + dt * v[p]



def main():
  for i in range(n_particles):
    x[i] = [random.random() * 0.4 + 0.2, random.random() * 0.4 + 0.2]
    v[i] = [1, 0]

  for f in range(100):
    for s in range(10):
      clear_grid()
      p2g()
      grid_op()
      g2p()
    scale = 20
    img = np.zeros(shape=(scale * n_grid, scale * n_grid))
    '''
    for i in range(scale * n_grid):
      for j in range(scale * n_grid):
        img[i, j] = grid_m[i // scale, j // scale] * 10
    print(v(0)[0])
    '''
    for i in range(n_particles):
      p_x = int(scale * x(0)[i] / dx)
      p_y = int(scale * x(1)[i] / dx)
      img[p_x, p_y] = 1
    cv2.imshow('MPM', img)
    cv2.waitKey(1)


if __name__ == '__main__':
  main()
