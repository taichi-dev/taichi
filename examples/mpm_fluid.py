import taichi as ti
import numpy as np
import random
import time

real = ti.f32
dim = 2
n_particles = 8192 * 4
n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2.0e-4
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho
E = 100

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

x, v = vec(), vec()
grid_v, grid_m = vec(), scalar()
C, J = mat(), scalar()

# ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda
# ti.cfg.verbose_kernel_launches = True
# ti.cfg.default_gpu_block_dim = 32

@ti.layout
def place():
  ti.root.dense(ti.k, n_particles).place(x, v, J, C)
  ti.root.dense(ti.ij, n_grid).place(grid_v, grid_m)


@ti.kernel
def clear_grid():
  for i, j in grid_m:
    grid_v[i, j] = [0, 0]
    grid_m[i, j] = 0


@ti.kernel
def p2g():
  for p in x:
    base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
    fx = x[p] * inv_dx - ti.cast(base, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1),
         0.5 * ti.sqr(fx - 0.5)]
    stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
    affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        offset = ti.Vector([i, j])
        dpos = (ti.cast(ti.Vector([i, j]), ti.f32) - fx) * dx
        weight = w[i](0) * w[j](1)
        grid_v[base + offset].atomic_add(weight * (p_mass * v[p] + affine @ dpos))
        grid_m[base + offset].atomic_add(weight * p_mass)


bound = 3


@ti.kernel
def grid_op():
  for i, j in grid_m:
    if grid_m[i, j] > 0:
      inv_m = 1 / grid_m[i, j]
      grid_v[i, j] = inv_m * grid_v[i, j]
      grid_v(1)[i, j] -= dt * 9.8
      if i < bound and grid_v(0)[i, j] < 0:
        grid_v(0)[i, j] = 0
      if i > n_grid - bound and grid_v(0)[i, j] > 0:
        grid_v(0)[i, j] = 0
      if j < bound and grid_v(1)[i, j] < 0:
        grid_v(1)[i, j] = 0
      if j > n_grid - bound and grid_v(1)[i, j] > 0:
        grid_v(1)[i, j] = 0


@ti.kernel
def g2p():
  for p in x:
    base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
    fx = x[p] * inv_dx - ti.cast(base, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
         0.5 * ti.sqr(fx - 0.5)]
    new_v = ti.Vector([0.0, 0.0])
    new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        dpos = ti.cast(ti.Vector([i, j]), ti.f32) - fx
        g_v = grid_v[base(0) + i, base(1) + j]
        weight = w[i](0) * w[j](1)
        new_v += weight * g_v
        new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

    v[p] = new_v
    x[p] += dt * v[p]
    J[p] *= 1 + dt * new_C.trace()
    C[p] = new_C

gui = ti.core.GUI("MPM", ti.veci(512, 512))
canvas = gui.get_canvas()

@ti.kernel
def copy_x(pos: ti.ext_arr()):
  for i in range(n_particles):
    pos[i * 2] = x[i][0]
    pos[i * 2 + 1] = x[i][1]

def main():
  for i in range(n_particles):
    x[i] = [random.random() * 0.4 + 0.2, random.random() * 0.4 + 0.2]
    v[i] = [0, -1]
    J[i] = 1

  for f in range(200):
    canvas.clear(0x112F41)
    t = time.time()
    for s in range(150):
      clear_grid()
      p2g()
      grid_op()
      g2p()
    print('{:.1f} ms per frame'.format(1000 * (time.time() - t)))

    pos = np.empty((2 * n_particles), dtype=np.float32)
    copy_x(pos)
    for i in range(n_particles):
      # canvas.circle(ti.vec(x[i][0], x[i][1])).radius(1).color(0x068587).finish()
      
      # Python binding here is still a bit slow...
      canvas.circle(ti.vec(pos[i * 2], pos[i * 2 + 1])).radius(1).color(0x068587).finish()
    gui.update()
  ti.profiler_print()

if __name__ == '__main__':
  main()
