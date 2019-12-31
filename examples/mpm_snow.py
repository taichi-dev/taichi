import taichi as ti
import random

dim = 2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1.0e-4
# p_vol = (dx * 0.5)**2
# p_rho = 1
# TODO: improve
p_mass = 1.0
p_vol = 1.0
# p_mass = p_vol * p_rho
harderning = 10
E = 1e2# 1e0
nu = 0.2
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1+nu) * (1 - 2 * nu))

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
F = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
J = ti.var(dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))

# ti.cfg.arch = ti.cuda
ti.get_runtime().print_preprocessed = True

@ti.kernel
def substep():
  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)]
    
    F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
    
    # e = ti.exp(harderning * (1.0 - J[p]))
    e = 1
    mu = mu_0*e
    la = lambda_0*e
    
    R, S = ti.polar_decompose(F[p], ti.f32)
    Jp = ti.determinant(F[p])
    stress = 2 * mu * (F[p] - R) @ ti.transposed(F[p]) + ti.Matrix.identity(ti.f32, 2) * la * Jp * (Jp - 1)
    # print(Jp)
    # stress = ti.Matrix.identity(ti.f32, 2) * (Jp - 1) * E
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * C[p]
    
    for i in ti.static(ti.ndrange(3, 3)):
      offset = ti.Vector([i, j])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]
      grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
      grid_m[base + offset] += weight * p_mass

  for i, j in grid_m:
    if grid_m[i, j] > 0:
      bound = 3
      inv_m = 1 / grid_m[i, j]
      grid_v[i, j] = inv_m * grid_v[i, j]
      grid_v[i, j][1] -= dt * 9.8
      if i < bound and grid_v[i, j][0] < 0:
        grid_v[i, j][0] = 0
      if i > n_grid - bound and grid_v[i, j][0] > 0:
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
    C[p] = new_C


gui = ti.core.GUI("MPM88", ti.veci(512, 512))
canvas = gui.get_canvas()

for i in range(n_particles):
  x[i] = [random.random() * 0.4 + 0.2, random.random() * 0.4 + 0.2]
  v[i] = [0, -1]
  F[i] = [[1, 0], [0, 1]]
  J[i] = 1

for frame in range(200):
  for s in range(50):
    grid_v.fill([0, 0])
    grid_m.fill(0)
    substep()

  canvas.clear(0x112F41)
  pos = x.to_numpy(as_vector=True)
  for i in range(n_particles):
    canvas.circle(ti.vec(pos[i, 0],
                         pos[i, 1])).radius(1.5).color(0x068587).finish()
  gui.update()
