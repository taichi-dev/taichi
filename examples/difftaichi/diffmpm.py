import taichi as ti
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import taichi as tc

real = ti.f32
ti.set_default_fp(real)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

actuator_id = ti.global_var(ti.i32)
particle_type = ti.global_var(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4

# ti.cfg.arch = ti.cuda


@ti.layout
def place():
  ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
  ti.root.dense(ti.i, n_actuators).place(bias)

  ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
  ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
  ti.root.dense(ti.l, max_steps).dense(ti.k, n_particles).place(x, v, C, F)
  ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
  ti.root.place(loss, x_avg)

  ti.root.lazy_grad()


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
def clear_actuation_grad():
  for t, i in actuation:
    actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
  for p in range(0, n_particles):
    base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
    fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)]
    new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
    J = ti.determinant(new_F)
    if particle_type[p] == 0:  # fluid
      sqrtJ = ti.sqrt(J)
      new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

    F[f + 1, p] = new_F
    r, s = ti.polar_decompose(new_F)

    act_id = actuator_id[p]

    act = actuation[f, ti.max(0, act_id)] * act_strength
    if act_id == -1:
      act = 0.0
    # ti.print(act)

    A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
    cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
    mass = 0.0
    if particle_type[p] == 0:
      mass = 4
      cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
    else:
      mass = 1
      cauchy = 2 * mu * (new_F - r) @ ti.transposed(new_F) + \
               ti.Matrix.diag(2, la * (J - 1) * J)
    cauchy += new_F @ A @ ti.transposed(new_F)
    stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
    affine = stress + mass * C[f, p]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        offset = ti.Vector([i, j])
        dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
        weight = w[i](0) * w[j](1)
        grid_v_in[base + offset].atomic_add(
            weight * (mass * v[f, p] + affine @ dpos))
        grid_m_in[base + offset].atomic_add(weight * mass)


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
  for i, j in grid_m_in:
    inv_m = 1 / (grid_m_in[i, j] + 1e-10)
    v_out = inv_m * grid_v_in[i, j]
    v_out[1] -= dt * gravity
    if i < bound and v_out[0] < 0:
      v_out[0] = 0
      v_out[1] = 0
    if i > n_grid - bound and v_out[0] > 0:
      v_out[0] = 0
      v_out[1] = 0
    if j < bound and v_out[1] < 0:
      v_out[0] = 0
      v_out[1] = 0
      normal = ti.Vector([0.0, 1.0])
      lsq = ti.sqr(normal).sum()
      if lsq > 0.5:
        if ti.static(coeff < 0):
          v_out(0).val = 0
          v_out(1).val = 0
        else:
          lin = (ti.transposed(v_out) @ normal)(0)
          if lin < 0:
            vit = v_out - lin * normal
            lit = vit.norm() + 1e-10
            if lit + coeff * lin <= 0:
              v_out(0).val = 0
              v_out(1).val = 0
            else:
              v_out = (1 + coeff * lin / lit) * vit
    if j > n_grid - bound and v_out[1] > 0:
      v_out[0] = 0
      v_out[1] = 0

    grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
  for p in range(0, n_particles):
    base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
    fx = x[f, p] * inv_dx - ti.cast(base, real)
    w = [
        0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)
    ]
    new_v = ti.Vector([0.0, 0.0])
    new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        dpos = ti.cast(ti.Vector([i, j]), real) - fx
        g_v = grid_v_out[base(0) + i, base(1) + j]
        weight = w[i](0) * w[j](1)
        new_v += weight * g_v
        new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

    v[f + 1, p] = new_v
    x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
    C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
  for i in range(n_actuators):
    act = 0.0
    for j in ti.static(range(n_sin_waves)):
      act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                    2 * math.pi / n_sin_waves * j)
    act += bias[i]
    actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
  for i in range(n_particles):
    contrib = 0.0
    if particle_type[i] == 1:
      contrib = 1.0 / n_solid_particles
    x_avg[None].atomic_add(contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
  dist = x_avg[None][0]
  loss[None] = -dist


@ti.complex_kernel
def advance(s):
  clear_grid()
  compute_actuation(s)
  p2g(s)
  grid_op()
  g2p(s)


@ti.complex_kernel_grad(advance)
def advance_grad(s):
  clear_grid()
  p2g(s)
  grid_op()

  g2p.grad(s)
  grid_op.grad()
  p2g.grad(s)
  compute_actuation.grad(s)


def forward(total_steps=steps):
  # simulation
  for s in range(total_steps - 1):
    advance(s)
  x_avg[None] = [0, 0]
  compute_x_avg()
  compute_loss()


class Scene:

  def __init__(self):
    self.n_particles = 0
    self.n_solid_particles = 0
    self.x = []
    self.actuator_id = []
    self.particle_type = []
    self.offset_x = 0
    self.offset_y = 0

  def add_rect(self, x, y, w, h, actuation, ptype=1):
    if ptype == 0:
      assert actuation == -1
    global n_particles
    w_count = int(w / dx) * 2
    h_count = int(h / dx) * 2
    real_dx = w / w_count
    real_dy = h / h_count
    for i in range(w_count):
      for j in range(h_count):
        self.x.append([
            x + (i + 0.5) * real_dx + self.offset_x,
            y + (j + 0.5) * real_dy + self.offset_y
        ])
        self.actuator_id.append(actuation)
        self.particle_type.append(ptype)
        self.n_particles += 1
        self.n_solid_particles += int(ptype == 1)

  def set_offset(self, x, y):
    self.offset_x = x
    self.offset_y = y

  def finalize(self):
    global n_particles, n_solid_particles
    n_particles = self.n_particles
    n_solid_particles = self.n_solid_particles
    print('n_particles', n_particles)
    print('n_solid', n_solid_particles)

  def set_n_actuators(self, n_act):
    global n_actuators
    n_actuators = n_act


def fish(scene):
  scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
  scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
  scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
  scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
  scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
  scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
  scene.set_n_actuators(4)


def robot(scene):
  scene.set_offset(0.1, 0.03)
  scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
  scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
  scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
  scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
  scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
  scene.set_n_actuators(4)


from renderer_vector import rgb_to_hex

gui = tc.core.GUI("Differentiable MPM", tc.veci(1024, 1024))
canvas = gui.get_canvas()


def visualize(s, folder):
  canvas.clear(0xFFFFFF)
  vec = tc.vec
  for i in range(n_particles):
    color = 0x111111
    aid = actuator_id[i]
    if aid != -1:
      act = actuation[s - 1, aid]
      color = rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
    canvas.circle(vec(x[s, i][0], x[s, i][1])).radius(2).color(color).finish()
  canvas.path(tc.vec(0.05, 0.02), tc.vec(0.95,
                                         0.02)).radius(3).color(0x0).finish()
  gui.update()

  os.makedirs(folder, exist_ok=True)
  gui.screenshot('{}/{:04d}.png'.format(folder, s))


def main():
  tc.set_gdb_trigger()
  # initialization
  scene = Scene()
  # fish(scene)
  robot(scene)
  scene.finalize()

  for i in range(n_actuators):
    for j in range(n_sin_waves):
      weights[i, j] = np.random.randn() * 0.01

  for i in range(scene.n_particles):
    x[0, i] = scene.x[i]
    F[0, i] = [[1, 0], [0, 1]]
    actuator_id[i] = scene.actuator_id[i]
    particle_type[i] = scene.particle_type[i]

  losses = []
  for iter in range(100):
    with ti.Tape(loss):
      forward()
    l = loss[None]
    losses.append(l)
    print('i=', iter, 'loss=', l)
    learning_rate = 0.1

    for i in range(n_actuators):
      for j in range(n_sin_waves):
        # print(weights.grad[i, j])
        weights[i, j] -= learning_rate * weights.grad[i, j]
      bias[i] -= learning_rate * bias.grad[i]

    if iter % 10 == 9:
      # visualize
      forward(1500)
      for s in range(63, 1500, 16):
        visualize(s, 'diffmpm/iter{:03d}/'.format(iter))

  # ti.profiler_print()
  plt.title("Optimization of Initial Velocity")
  plt.ylabel("Loss")
  plt.xlabel("Gradient Descent Iterations")
  plt.plot(losses)
  plt.show()


if __name__ == '__main__':
  main()
