import taichi as ti
import random

dim = 2
n_particles, n_grid = 9000, 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-4
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
harderning = 10
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
F = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
material = ti.var(dt=ti.i32, shape=n_particles)
Jp = ti.var(dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))

ti.cfg.arch = ti.cuda

@ti.kernel
def substep():
  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)]
    F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p] # deformation gradient update
    e = ti.exp(harderning * (1.0 - Jp[p]))
    mu, la = mu_0 * e, lambda_0 * e
    
    if material[p] == 0:
      mu = 0.0
    
    U, sig, V = ti.svd(F[p])
    
    J = 1.0
    for d in ti.static(range(2)):
      new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 7.5e-3)
      Jp[p] *= sig[d, d] / new_sig
      sig[d, d] = new_sig
      J *= new_sig
    
    F[p] = U @ sig @ V.T()
    stress = 2 * mu * (F[p] - U @ V.T()) @ F[p].T() + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * C[p]
    
    for i, j in ti.static(ti.ndrange(3, 3)):
      offset = ti.Vector([i, j])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]
      grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
      grid_m[base + offset] += weight * p_mass

  for i, j in grid_m:
    if grid_m[i, j] > 0:
      inv_m = 1 / grid_m[i, j]
      grid_v[i, j] = inv_m * grid_v[i, j]
      grid_v[i, j][1] -= dt * 9.8
      if i < 3 and grid_v[i, j][0] < 0:          grid_v[i, j][0] = 0
      if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
      if j < 3 and grid_v[i, j][1] < 0:          grid_v[i, j][1] = 0
      if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)]
    new_v = ti.Vector.zero(ti.f32, 2)
    new_C = ti.Matrix.zero(ti.f32, 2, 2)
    for i, j in ti.static(ti.ndrange(3, 3)):
      dpos = ti.Vector([i, j]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j])]
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
      new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx
    v[p], C[p] = new_v, new_C
    x[p] += dt * v[p]

gui = ti.core.GUI("MPM88", ti.veci(512, 512))
canvas = gui.get_canvas()

for i in range(n_particles):
  x[i] = [random.random() * 0.1 + 0.2 + 0.1 * (i // 3000), random.random() * 0.4 + 0.4]
  material[i] = (i // 3000)
  v[i] = [0, -3]
  F[i] = [[1, 0], [0, 1]]
  Jp[i] = 1

for frame in range(20000):
  for s in range(100):
    grid_v.fill([0, 0])
    grid_m.fill(0)
    substep()

  canvas.clear(0x112F41)
  pos = x.to_numpy(as_vector=True)
  for i in range(n_particles):
    canvas.circle(ti.vec(pos[i, 0],
                         pos[i, 1])).radius(1.5).color(0x068587).finish()
  gui.update()
  # gui.screenshot(f"{frame:05d}.png")
