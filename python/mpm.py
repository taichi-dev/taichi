import taichi_lang as ti
import numpy as np
import random
import cv2

real = ti.f32
dim = 2
n_particles = 8192
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
def p2g():
  for p in x(0):
    base_coord = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
    fx = x[p] * inv_dx - ti.cast(base_coord, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 * ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        # dpos = (ti.Vector([i, j]) - fx) * dx
        weight = w[i](0) * w[j](1)
        #grid_v[base_coord(0) + i, base_coord(1) + j].assign(
        #  grid_v[base_coord(0) + i, base_coord(1) + j] + weight * p_mass * v[p])
        #grid_m[base_coord(0) + i, base_coord(1) + j].assign(
        #  grid_m[base_coord(0) + i, base_coord(1) + j] + weight * p_mass)
        grid_m[base_coord(0) + i, base_coord(1) + j].assign(1.0)


'''
@ti.kernel
def grid_op():
  pass

@ti.kernel
def g2p():
  pass
'''


def main():
  pass


if __name__ == '__main__':
  for i in range(n_particles):
    x[i] = [random.random() * 0.4 + 0.2, random.random() * 0.4 + 0.2]
    v[i] = [1, 1]

  for f in range(100):
    for s in range(1):
      p2g()
      # grid_op()
      # g2p()
    scale = 20
    img = np.zeros(shape=(scale * n_grid, scale * n_grid))
    for i in range(n_particles):
      p_x = int(scale * x(0)[i] / dx)
      p_y = int(scale * x(1)[i] / dx)
      img[p_x, p_y] = 1
    cv2.imshow('MPM', img)
    cv2.waitKey(1)
