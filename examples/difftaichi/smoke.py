import taichi as ti
import time
import numpy as np
import cv2
import os
from imageio import imread, imwrite

real = ti.f32
ti.set_default_fp(real)

num_iterations = 50
n_grid = 110
dx = 1.0 / n_grid
num_iterations_gauss_seidel = 10
p_dims = num_iterations_gauss_seidel + 1
steps = 100
learning_rate = 100

scalar = lambda: ti.var(dt=real)
vector = lambda: ti.Vector(2, dt=real)

v = vector()
div = scalar()
p = scalar()
v_updated = vector()
target = scalar()
smoke = scalar()
loss = scalar()

ti.cfg.arch = ti.cuda

@ti.layout
def place():
  ti.root.dense(ti.l, steps * p_dims).dense(ti.ij, n_grid).place(p)
  ti.root.dense(ti.l, steps).dense(ti.ij, n_grid).place(v, v_updated, smoke, div)
  ti.root.dense(ti.ij, n_grid).place(target)
  ti.root.place(loss)
  ti.root.lazy_grad()


# Integer modulo operator for positive values of n
@ti.func
def imod(n, divisor):
  ret = 0
  if n > 0:
    ret = n - divisor * (n // divisor)
  else:
    ret = divisor + n - divisor * (-n // divisor)
  return ret

@ti.func
def dec_index(index):
  new_index = index - 1
  if new_index < 0:
    new_index = n_grid - 1
  return new_index


@ti.func
def inc_index(index):
  new_index = index + 1
  if new_index >= n_grid:
    new_index = 0
  return new_index


@ti.kernel
def compute_div(t: ti.i32):
  for y in range(n_grid):
    for x in range(n_grid):
      div[t, y, x] = -0.5 * dx * (
            v_updated[t, inc_index(y), x][0] - v_updated[t, dec_index(y), x][0] +
            v_updated[t, y, inc_index(x)][1] - v_updated[t, y, dec_index(x)][1])


@ti.kernel
def compute_p(t: ti.i32, k: ti.i32):
  for y in range(n_grid):
    for x in range(n_grid):
      a = k + t * num_iterations_gauss_seidel
      p[a + 1, y, x] = (div[t, y, x] + p[a, dec_index(y), x] + p[
        a, inc_index(y), x] + p[a, y, dec_index(x)] + p[
                          a, y, inc_index(x)]) / 4.0


@ti.kernel
def update_v(t: ti.i32):
  for y in range(n_grid):
    for x in range(n_grid):
      a = num_iterations_gauss_seidel * t - 1
      v[t, y, x][0] = v_updated[t, y, x][0] - 0.5 * (
            p[a, inc_index(y), x] - p[a, dec_index(y), x]) / dx
      v[t, y, x][1] = v_updated[t, y, x][1] - 0.5 * (
            p[a, y, inc_index(x)] - p[a, y, dec_index(x)]) / dx


def advect(field, field_out, t_offset):
  @ti.kernel
  def kernel(t: ti.i32):
    """Move field smoke according to x and y velocities (vx and vy)
       using an implicit Euler integrator."""
    for y in range(n_grid):
      for x in range(n_grid):
        center_x = y - v[t + t_offset, y, x][0]
        center_y = x - v[t + t_offset, y, x][1]
        
        # Compute indices of source cell
        left_ix = ti.cast(ti.floor(center_x), ti.i32)
        top_ix = ti.cast(ti.floor(center_y), ti.i32)
        
        rw = center_x - left_ix  # Relative weight of right-hand cell
        bw = center_y - top_ix  # Relative weight of bottom cell
        
        # Wrap around edges
        # TODO: implement mod (%) operator
        left_ix = imod(left_ix, n_grid)
        right_ix = left_ix + 1
        right_ix = imod(right_ix, n_grid)
        top_ix = imod(top_ix, n_grid)
        bot_ix = top_ix + 1
        bot_ix = imod(bot_ix, n_grid)
        
        # Linearly-weighted sum of the 4 surrounding cells
        field_out[t, y, x] = (1 - rw) * (
              (1 - bw) * field[t - 1, left_ix, top_ix] + bw * field[
            t - 1, left_ix, bot_ix]) + rw * (
                               (1 - bw) * field[t - 1, right_ix, top_ix] + bw *
                               field[t - 1, right_ix, bot_ix])
  kernel.materialize()
  kernel.grad.materialize()
  return kernel

@ti.kernel
def compute_loss():
  for i in range(n_grid):
    for j in range(n_grid):
      ti.atomic_add(loss, ti.sqr(target[i, j] - smoke[steps - 1, i, j]) * (
            1 / n_grid ** 2))


@ti.kernel
def apply_grad():
  # gradient descent
  for i in range(n_grid):
    for j in range(n_grid):
      v[0, i, j] -= learning_rate * v.grad[0, i, j]

advect_v = advect(v, v_updated, -1)
advect_smoke = advect(smoke, smoke, 0)

def forward(output=None):
  T = time.time()
  for t in range(1, steps):
    advect_v(t)
    
    compute_div(t)
    for k in range(num_iterations_gauss_seidel):
      compute_p(t, k)
    
    update_v(t)
    advect_smoke(t)
    
    if output:
      os.makedirs(output, exist_ok=True)
      smoke_ = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
      for i in range(n_grid):
        for j in range(n_grid):
          smoke_[i, j] = smoke[t, i, j]
      cv2.imshow('smoke', smoke_)
      cv2.waitKey(1)
      cv2.imwrite("{}/{:04d}.png".format(output, t), 255 * smoke_)
  compute_loss()
  print('forward time', (time.time() - T) * 1000, 'ms')


def main():
  print("Loading initial and target states...")
  initial_smoke_img = imread("init_smoke.png")[:, :, 0] / 255.0
  target_img = imread("peace.png")[::2, ::2, 3] / 255.0
  
  for i in range(n_grid):
    for j in range(n_grid):
      target[i, j] = target_img[i, j]
      smoke[0, i, j] = initial_smoke_img[i, j]
  
  for opt in range(num_iterations):
    t = time.time()
    with ti.Tape(loss):
      output = "test" if opt % 10 == 0 else None
      forward(output)
    print('total time', (time.time() - t) * 1000, 'ms')
    
    print('Iter', opt, ' Loss =', loss[None])
    apply_grad()
  
  forward("output")


if __name__ == '__main__':
  main()
