import taichi as ti
import random

real = ti.f32
dim = 2
n_particle_x = 64
n_particle_y = 8
n_particles = 8192 * 4
n_elements = (n_particle_x - 1) * (n_particle_y - 1) * 2
n_grid = 64
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-4
p_mass = 1
p_vol = 1
mu = 1
la = 1

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

x, v = vec(), vec()
grid_v, grid_m = vec(), scalar()
C, J = mat(), scalar()
restT = mat()
total_energy = scalar()
vertices = ti.var(ti.i32)

ti.cfg.arch = ti.cuda

@ti.layout
def place():
  ti.root.dense(ti.k, n_particles).place(x, x.grad, v, J, C)
  ti.root.dense(ti.ij, n_grid).place(grid_v, grid_m)
  ti.root.dense(ti.i, n_elements).place(restT, restT.grad)
  ti.root.dense(ti.ij, (n_elements, 3)).place(vertices)
  ti.root.place(total_energy, total_energy.grad)


@ti.kernel
def clear_grid():
  for i, j in grid_m:
    grid_v[i, j] = [0, 0]
    grid_m[i, j] = 0

@ti.func
def compute_T(i):
  a = vertices[i, 0]
  b = vertices[i, 1]
  c = vertices[i, 2]
  ab = x[b] - x[a]
  ac = x[c] - x[a]
  return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])

@ti.kernel
def compute_rest_T():
  for i in range(n_elements):
    restT[i] = compute_T(i)

@ti.kernel
def compute_total_energy():
  for i in range(n_elements):
    currentT = compute_T(i)
    F = restT[i] * currentT.inverse()
    # Fixed Corotated
    I1 = (F @ F.transpose()).trace()
    J = F.determinant()
    element_energy = 0.5 * mu * (I1 - 3) - mu * ti.log(J) + 0.5 * la * ti.log(J) ** 2
    ti.atomic_add(total_energy[None], element_energy)


@ti.kernel
def p2g():
  for p in x:
    base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
    fx = x[p] * inv_dx - ti.cast(base, ti.f32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1),
         0.5 * ti.sqr(fx - 0.5)]
    affine = p_mass * C[p]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        offset = ti.Vector([i, j])
        dpos = (ti.cast(ti.Vector([i, j]), ti.f32) - fx) * dx
        weight = w[i](0) * w[j](1)
        grid_v[base + offset].atomic_add(weight * (p_mass * v[p] + x.grad[p] + affine @ dpos))
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

def main():
  for i in range(n_particle_x):
    for j in range(n_particle_y):
      t = i * n_particle_y + j
      x[t] = [0.2 + i * dx * 0.5, 0.6 + j * dx * 0.5]
      v[t] = [0, 0]
      J[t] = 1

  for f in range(200):
    canvas.clear(0x112F41)
    for s in range(150):
      clear_grid()
      p2g()
      grid_op()
      g2p()


    # TODO: why is visualization so slow?
    for i in range(n_particles):
      canvas.circle(ti.vec(x[i][0], x[i][1])).radius(1).color(0x068587).finish()
    gui.update()
  ti.profiler_print()

if __name__ == '__main__':
  main()
