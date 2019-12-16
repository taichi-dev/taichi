import taichi as ti
from mpl_toolkits.mplot3d import Axes3D
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import taichi as tc

real = ti.f32
ti.set_default_fp(real)

dim = 3
# this will be overwritten
n_particles = 0
n_solid_particles = 0
n_actuators = 0
n_grid = 64
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 512
steps = 512
gravity = 10
target = [0.8, 0.2, 0.2]
use_apic = False

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

actuator_id = ti.global_var(ti.i32)
particle_type = ti.global_var(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

screen = ti.Vector(3, dt=real)

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 40
act_strength = 5

# ti.cfg.arch = ti.x86_64
# ti.cfg.use_llvm = True
ti.cfg.arch = ti.cuda

# ti.cfg.print_ir = True

visualize_resolution = 256


@ti.layout
def place():
  ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
  ti.root.dense(ti.i, n_actuators).place(bias)

  ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
  ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
  ti.root.dense(ti.l, max_steps).dense(ti.k, n_particles).place(x, v, C, F)
  ti.root.dense(ti.ijk, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
  ti.root.place(loss, x_avg)
  ti.root.dense(ti.ij,
                (visualize_resolution, visualize_resolution)).place(screen)

  ti.root.lazy_grad()


def zero_vec():
  return [0.0, 0.0, 0.0]


def zero_matrix():
  return [zero_vec(), zero_vec(), zero_vec()]


@ti.kernel
def clear_grid():
  for i, j, k in grid_m_in:
    grid_v_in[i, j, k] = [0, 0, 0]
    grid_m_in[i, j, k] = 0
    grid_v_in.grad[i, j, k] = [0, 0, 0]
    grid_m_in.grad[i, j, k] = 0
    grid_v_out.grad[i, j, k] = [0, 0, 0]


@ti.kernel
def clear_particle_grad():
  # for all time steps and all particles
  for f, i in x:
    x.grad[f, i] = zero_vec()
    v.grad[f, i] = zero_vec()
    C.grad[f, i] = zero_matrix()
    F.grad[f, i] = zero_matrix()


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
    new_F = (ti.Matrix.diag(dim=dim, val=1) + dt * C[f, p]) @ F[f, p]
    J = ti.determinant(new_F)
    if particle_type[p] == 0:  # fluid
      sqrtJ = ti.sqrt(J)
      # TODO: need pow(x, 1/3)
      new_F = ti.Matrix([[sqrtJ, 0, 0], [0, sqrtJ, 0], [0, 0, 1]])

    F[f + 1, p] = new_F
    # r, s = ti.polar_decompose(new_F)

    act_id = actuator_id[p]

    act = actuation[f, ti.max(0, act_id)] * act_strength
    if act_id == -1:
      act = 0.0
    # ti.print(act)

    A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) * act
    cauchy = ti.Matrix(zero_matrix())
    mass = 0.0
    ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if particle_type[p] == 0:
      mass = 4
      cauchy = ti.Matrix(ident) * (J - 1) * E
    else:
      mass = 1
      cauchy = mu * (new_F @ ti.transposed(new_F)) + ti.Matrix(ident) * (
          la * ti.log(J) - mu)
    cauchy += new_F @ A @ ti.transposed(new_F)
    stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
    affine = stress + mass * C[f, p]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        for k in ti.static(range(3)):
          offset = ti.Vector([i, j, k])
          dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
          weight = w[i](0) * w[j](1) * w[k](2)
          grid_v_in[base + offset].atomic_add(
              weight * (mass * v[f, p] + affine @ dpos))
          grid_m_in[base + offset].atomic_add(weight * mass)


bound = 3
coeff = 1.5


@ti.kernel
def grid_op():
  for i, j, k in grid_m_in:
    inv_m = 1 / (grid_m_in[i, j, k] + 1e-10)
    v_out = inv_m * grid_v_in[i, j, k]
    v_out[1] -= dt * gravity

    if i < bound and v_out[0] < 0:
      v_out[0] = 0
      v_out[1] = 0
      v_out[2] = 0
    if i > n_grid - bound and v_out[0] > 0:
      v_out[0] = 0
      v_out[1] = 0
      v_out[2] = 0

    if k < bound and v_out[2] < 0:
      v_out[0] = 0
      v_out[1] = 0
      v_out[2] = 0
    if k > n_grid - bound and v_out[2] > 0:
      v_out[0] = 0
      v_out[1] = 0
      v_out[2] = 0

    if j < bound and v_out[1] < 0:
      v_out[0] = 0
      v_out[1] = 0
      v_out[2] = 0
      normal = ti.Vector([0.0, 1.0, 0.0])
      lsq = ti.sqr(normal).sum()
      if lsq > 0.5:
        if ti.static(coeff < 0):
          v_out[0] = 0
          v_out[1] = 0
          v_out[2] = 0
        else:
          lin = (ti.transposed(v_out) @ normal)(0)
          if lin < 0:
            vit = v_out - lin * normal
            lit = vit.norm() + 1e-10
            if lit + coeff * lin <= 0:
              v_out[0] = 0
              v_out[1] = 0
              v_out[2] = 0
            else:
              v_out = (1 + coeff * lin / lit) * vit
    if j > n_grid - bound and v_out[1] > 0:
      v_out[0] = 0
      v_out[1] = 0
      v_out[2] = 0

    grid_v_out[i, j, k] = v_out


@ti.kernel
def g2p(f: ti.i32):
  for p in range(0, n_particles):
    base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
    fx = x[f, p] * inv_dx - ti.cast(base, real)
    w = [
        0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)
    ]
    new_v = ti.Vector(zero_vec())
    new_C = ti.Matrix(zero_matrix())

    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        for k in ti.static(range(3)):
          dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
          g_v = grid_v_out[base(0) + i, base(1) + j, base(2) + k]
          weight = w[i](0) * w[j](1) * w[k](2)
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


def forward(total_steps=steps):
  # simulation
  for s in range(total_steps - 1):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)

  x_avg[None] = [0, 0, 0]
  compute_x_avg()
  compute_loss()
  return loss[None]


def backward():
  clear_particle_grad()

  compute_loss.grad()
  compute_x_avg.grad()
  for s in reversed(range(steps - 1)):
    # Since we do not store the grid history (to save space), we redo p2g and grid op
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


class Scene:

  def __init__(self):
    self.n_particles = 0
    self.n_solid_particles = 0
    self.x = []
    self.actuator_id = []
    self.particle_type = []
    self.offset_x = 0
    self.offset_y = 0
    self.offset_z = 0
    self.num_actuators = 0

  def new_actuator(self):
    self.num_actuators += 1
    global n_actuators
    n_actuators = self.num_actuators
    return self.num_actuators - 1

  def add_rect(self, x, y, z, w, h, d, actuation, ptype=1):
    if ptype == 0:
      assert actuation == -1
    global n_particles
    density = 3
    w_count = int(w / dx * density)
    h_count = int(h / dx * density)
    d_count = int(d / dx * density)
    real_dx = w / w_count
    real_dy = h / h_count
    real_dz = d / d_count

    if ptype == 1:
      for i in range(w_count):
        for j in range(h_count):
          for k in range(d_count):
            self.x.append([
                x + (i + 0.5) * real_dx + self.offset_x,
                y + (j + 0.5) * real_dy + self.offset_y,
                z + (k + 0.5) * real_dz + self.offset_z
            ])
            self.actuator_id.append(actuation)
            self.particle_type.append(ptype)
            self.n_particles += 1
            self.n_solid_particles += int(ptype == 1)
            if self.n_particles % 1000 == 0:
              print("num particles", self.n_particles)
    else:
      for i in range(w_count):
        for j in range(h_count):
          for k in range(d_count):
            self.x.append([
                x + random.random() * w + self.offset_x,
                y + random.random() * h + self.offset_y,
                z + random.random() * d + self.offset_z
            ])
            self.actuator_id.append(actuation)
            self.particle_type.append(ptype)
            self.n_particles += 1
            self.n_solid_particles += int(ptype == 1)
            if self.n_particles % 1000 == 0:
              print("num particles", self.n_particles)

  def set_offset(self, x, y, z):
    self.offset_x = x
    self.offset_y = y
    self.offset_z = z

  def finalize(self):
    global n_particles, n_solid_particles
    n_particles = self.n_particles
    n_solid_particles = max(self.n_solid_particles, 1)
    print('n_particles', n_particles)
    print('n_solid', n_solid_particles)

  def set_n_actuators(self, n_act):
    global n_actuators
    n_actuators = n_act


gui = tc.core.GUI("Differentiable MPM", tc.veci(1024, 1024))
canvas = gui.get_canvas()


@ti.kernel
def splat(t: ti.i32):
  for i in range(n_particles):
    pos = ti.cast(x[t, i] * visualize_resolution, ti.i32)
    screen[pos[0], pos[1]][0] += 0.1


res = [visualize_resolution, visualize_resolution]


@ti.kernel
def copy_back_and_clear(img: np.ndarray):
  for i in range(res[0]):
    for j in range(res[1]):
      coord = ((res[1] - 1 - j) * res[0] + i) * 3
      for c in ti.static(range(3)):
        img[coord + c] = screen[i, j][2 - c]
        screen[i, j][2 - c] = 0


def robot(scene):
  block_size = 0.1
  # scene.set_offset(0.1, 0.10, 0.3)
  scene.set_offset(0.1, 0.05, 0.3)

  def add_leg(x, y, z):
    for i in range(4):
      scene.add_rect(x + block_size / 2 * (i // 2),
                     y + 0.7 * block_size / 2 * (i % 2), z, block_size / 2,
                     0.7 * block_size / 2, block_size, scene.new_actuator())

  for i in range(4):
    add_leg(i // 2 * block_size * 2, 0.0, i % 2 * block_size * 2)
  for i in range(3):
    scene.add_rect(block_size * i, 0, block_size, block_size, block_size * 0.7,
                   block_size, -1, 1)


def main():
  tc.set_gdb_trigger()
  # initialization
  scene = Scene()
  # fish(scene)
  robot(scene)
  # scene.add_rect(0.4, 0.4, 0.2, 0.1, 0.3, 0.1, -1, 1)
  scene.finalize()

  for i in range(n_actuators):
    for j in range(n_sin_waves):
      weights[i, j] = np.random.randn() * 0.01

  for i in range(scene.n_particles):
    x[0, i] = scene.x[i]
    F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    actuator_id[i] = scene.actuator_id[i]
    particle_type[i] = scene.particle_type[i]

  fig = plt.figure()
  plt.ion()
  ax = fig.add_subplot(111, projection='3d')

  losses = []
  for iter in range(100):
    t = time.time()
    ti.clear_all_gradients()
    l = forward()
    losses.append(l)
    loss.grad[None] = 1
    backward()
    per_iter_time = time.time() - t
    print('i=', iter, 'loss=', l, F' per iter {per_iter_time:.2f}s')
    learning_rate = 30

    for i in range(n_actuators):
      for j in range(n_sin_waves):
        # print(weights.grad[i, j])
        weights[i, j] -= learning_rate * weights.grad[i, j]
      bias[i] -= learning_rate * bias.grad[i]

    if iter % 20 == 0 and iter > 0:
      # visualize
      forward()
      for s in range(7, steps, 2):
        '''
        print(s)
        img = np.zeros((res[1] * res[0] * 3,), dtype=np.float32)
        splat(s)
        copy_back_and_clear(img)
        img = img.reshape(res[1], res[0], 3)
        img = np.sqrt(img)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        '''
        '''
        xs, ys, zs = [], [], []
        aas, bs, cs = [], [], []
        for i in range(n_particles):
          if particle_type[i] == 0:
            xs.append(x[s, i][0])
            ys.append(x[s, i][2])
            zs.append(x[s, i][1])
          else:
            aas.append(x[s, i][0])
            bs.append(x[s, i][2])
            cs.append(x[s, i][1])

        ax.scatter(aas, bs, cs, marker='o')
        ax.scatter(xs, ys, zs, marker='o')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        plt.draw()
        plt.pause(0.001)
        plt.cla()
        '''

        def to255(x):
          return int(max(min(x * 255, 255), 0))

        xs, ys, zs = [], [], []
        us, vs, ws = [], [], []
        cs = []
        folder = 'mpm3d/iter{:04d}/'.format(iter)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_particles):
          xs.append(x[s, i][0])
          ys.append(x[s, i][1])
          zs.append(x[s, i][2])
          us.append(v[s, i][0])
          vs.append(v[s, i][1])
          ws.append(v[s, i][2])

          if particle_type[i] == 0:
            # fluid
            r = 0.3
            g = 0.3
            b = 1.0
          else:
            # neohookean
            if actuator_id[i] != -1:
              # actuated
              act = actuation[s, actuator_id[i]] * 0.5
              r = 0.5 - act
              g = 0.5 - abs(act)
              b = 0.5 + act
            else:
              r, g, b = 0.4, 0.4, 0.4

          color = to255(r) * 65536 + 256 * to255(g) + to255(b)
          cs.append(color)
        data = np.array(xs + ys + zs + us + vs + ws + cs, dtype=np.float32)
        data.tofile(open('{}/{:04}.bin'.format(folder, s), 'wb'))

  # ti.profiler_print()
  plt.title("Optimization of Initial Velocity")
  plt.ylabel("Loss")
  plt.xlabel("Gradient Descent Iterations")
  plt.plot(losses)
  plt.show()


if __name__ == '__main__':
  main()
