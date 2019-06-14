import taichi_lang as ti
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

real = ti.f32
dim = 2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
p_mass = 1
p_vol = 1
E = 100
# TODO: update
mu = E
la = E
max_steps = 32
steps = 32
assert steps <= max_steps
gravity = 9.8
target = [0.3, 0.6]

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

x, v, x_avg = vec(), vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

init_v = vec()
loss = scalar()

# ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda
# ti.cfg.print_ir = True


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.k, n_particles).place(x, v, C, F).place(
    x.grad, v.grad, C.grad, F.grad)
  ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out).place(
    grid_v_in.grad, grid_m_in.grad, grid_v_out.grad)
  ti.root.place(init_v, init_v.grad, loss, loss.grad, x_avg, x_avg.grad)


@ti.kernel
def p2g(f: ti.i32):
  for p in range(0, n_particles):
    ti.print(f)
    base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
    fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1),
         0.5 * ti.sqr(fx - 0.5)]
    new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
    F[f + 1, p] = new_F


def main():
  init_v[None] = [0, 0]

  for i in range(n_particles):
    x[0, i] = [random.random() * 0.4 + 0.3, random.random() * 0.4 + 0.3]
    F[0, i] = [[1, 0], [0, 1]]

  p2g(0)
  print("are")
  exit(0)


if __name__ == '__main__':
  main()
