import taichi as ti
import random

dim = 2
n_particles = 8192; n_grid = 128
dx = 1 / n_grid; inv_dx = 1 / dx
dt = 2.0e-4
p_vol = (dx * 0.5) ** 2
p_rho = 1; p_mass = p_vol * p_rho
E = 400

scalar = lambda: ti.var(dt=ti.f32)
vec = lambda: ti.Vector(dim, dt=ti.f32)
mat = lambda: ti.Matrix(dim, dim, dt=ti.f32)

x, v = vec(), vec()
grid_v, grid_m = vec(), scalar()
C, J = mat(), scalar()

ti.cfg.arch = ti.cuda

@ti.layout
def place():
  ti.root.dense(ti.k, n_particles).place(x, v, J, C)
  ti.root.dense(ti.ij, n_grid).place(grid_v, grid_m)

@ti.kernel
def substep():
  for i, j in grid_v:
    grid_v[i, j] = ti.Matrix.zero(ti.f32, 2)
    grid_m[i, j] = 0

  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)]
    stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
    affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        offset = ti.Vector([i, j])
        dpos = (offset.cast(float) - fx) * dx
        weight = w[i][0] * w[j][1]
        grid_v[base + offset].atomic_add(weight * (p_mass * v[p] + affine @ dpos))
        grid_m[base + offset].atomic_add(weight * p_mass)

  for i, j in grid_m:
    if grid_m[i, j] > 0:
      bound = 3
      inv_m = 1 / grid_m[i, j]
      grid_v[i, j] = inv_m * grid_v[i, j]
      grid_v[i, j][1] -= dt * 9.8
      if i < bound and grid_v[i, j][0] < 0:
        grid_v[i, j][0] = 0
      if i > n_grid - bound and grid_v(0)[i, j] > 0:
        grid_v[i, j][0] = 0
      if j < bound and grid_v[i, j][1] < 0:
        grid_v[i, j][1] = 0
      if j > n_grid - bound and grid_v[i, j][1] > 0:
        grid_v[i, j][1] = 0

  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)]

    new_v = ti.Vector.zero(ti.f32, 2)
    new_C = ti.Matrix.zero(ti.f32, 2, 2)

    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        dpos = ti.Vector([i, j]).cast(float) - fx
        g_v = grid_v[base + ti.Vector([i, j])]
        weight = w[i][0] * w[j][1]
        new_v += weight * g_v
        new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

    v[p] = new_v
    x[p] += dt * v[p]
    J[p] *= 1 + dt * new_C.trace()
    C[p] = new_C

gui = ti.core.GUI("MPM", ti.veci(512, 512))
canvas = gui.get_canvas()

for i in range(n_particles):
  x[i] = [random.random() * 0.4 + 0.2, random.random() * 0.4 + 0.2]
  v[i] = [0, -1]
  J[i] = 1

for f in range(200):
  canvas.clear(0x112F41)
  for s in range(50):
    substep()

  pos = x.to_numpy()
  for i in range(n_particles):
    canvas.circle(ti.vec(pos[i, 0, 0], pos[i, 1, 0])).radius(1.5).color(0x068587).finish()
  gui.update()
