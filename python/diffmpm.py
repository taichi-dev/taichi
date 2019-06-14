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
def set_v():
  for i in range(n_particles):
    v[0, i] = init_v

@ti.kernel
def clear_grid():
  for i, j in grid_m_in:
    grid_v_in[i, j] = [0, 0]
    grid_m_in[i, j] = 0
    grid_v_in.grad[i, j] = [0, 0]
    grid_m_in.grad[i, j] = 0
    grid_v_out.grad[i, j] = [0, 0]

@ti.kernel
def clear_particle_grad():
  # for all time steps and all particles
  for f, i in x:
    x.grad[f, i] = [0, 0]
    v.grad[f, i] = [0, 0]
    C.grad[f, i] = [[0, 0], [0, 0]]
    F.grad[f, i] = [[0, 0], [0, 0]]


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

    J = ti.determinant(new_F)
    r, s = ti.polar_decompose(new_F)
    cauchy = 2 * mu * (new_F - r) @ ti.transposed(new_F) + \
             ti.Matrix.diag(2, la * (J - 1) * J)
    stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
    affine = stress + p_mass * C[f, p]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        offset = ti.Vector([i, j])
        dpos = (ti.cast(ti.Vector([i, j]), ti.f32) - fx) * dx
        weight = w[i](0) * w[j](1)
        grid_v_in[base + offset].atomic_add(
          weight * (p_mass * v[f, p] + affine @ dpos))
        grid_m_in[base + offset].atomic_add(weight * p_mass)


bound = 3


@ti.kernel
def grid_op():
  for i, j in grid_m_in:
    v_out = ti.Vector([0.0, 0.0])
    #if grid_m_in[i, j] > 0:

    inv_m = 1 / (grid_m_in[i, j] + 1e-10)

    v_out = inv_m * grid_v_in[i, j]

    v_out(1).val -= dt * gravity

    if i < bound and v_out(0) < 0:
      v_out(0).val = 0
    if i > n_grid - bound and v_out(0) > 0:
      v_out(0).val = 0
    if j < bound and v_out(1) < 0:
      v_out(1).val = 0
    if j > n_grid - bound and v_out(1) > 0:
      v_out(1).val = 0
    grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
  for p in range(0, n_particles):
    base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
    fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
         0.5 * ti.sqr(fx - 0.5)]
    new_v = ti.Vector([0.0, 0.0])
    new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        dpos = ti.cast(ti.Vector([i, j]), ti.f32) - fx
        g_v = grid_v_out[base(0) + i, base(1) + j]
        weight = w[i](0) * w[j](1)
        new_v += weight * g_v
        new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

    v[f + 1, p] = new_v
    x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
    C[f + 1, p] = new_C


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
