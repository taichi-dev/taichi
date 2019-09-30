# TO run on CPU:
#   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3 smoke_pytorch.py

import torch
import time
import math
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import pdb
from imageio import imread, imwrite
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_iterations = 50
n_grid = 110
dx = 1.0 / n_grid
num_iterations_gauss_seidel = 10
p_dims = num_iterations_gauss_seidel + 1
steps = 100
learning_rate = 100

def roll_col(t, n):
  return torch.cat((t[:, -n:], t[:, :-n]), axis=1)

def roll_row(t, n):
  return torch.cat((t[-n:, :], t[:-n, :]), axis=0)

def project(vx, vy):
  """Project the velocity field to be approximately mass-conserving,
     using a few iterations of Gauss-Seidel."""
  p = torch.zeros(vx.shape).to(device)
  h = 1.0/vx.shape[0]
  div = -0.5 * h * (roll_row(vx, -1) - roll_row(vx, 1)
                  + roll_col(vy, -1) - roll_col(vy, 1))

  for k in range(10):
    p = (div + roll_row(p, 1) + roll_row(p, -1)
             + roll_col(p, 1) + roll_col(p, -1))/4.0

  vx -= 0.5*(roll_row(p, -1) - roll_row(p, 1))/h
  vy -= 0.5*(roll_col(p, -1) - roll_col(p, 1))/h
  return vx, vy

def advect(f, vx, vy):
  """Move field f according to x and y velocities (u and v)
     using an implicit Euler integrator."""
  rows, cols = f.shape
  cell_ys, cell_xs = torch.meshgrid(torch.arange(rows), torch.arange(cols))
  cell_ys = torch.transpose(cell_ys, 0, 1).float().to(device)
  cell_xs = torch.transpose(cell_xs, 0, 1).float().to(device)
  center_xs = (cell_xs - vx).flatten()
  center_ys = (cell_ys - vy).flatten()

  # Compute indices of source cells.
  left_ix = torch.floor(center_xs).long()
  top_ix  = torch.floor(center_ys).long()
  rw = center_xs - left_ix.float()      # Relative weight of right-hand cells.
  bw = center_ys - top_ix.float()       # Relative weight of bottom cells.
  left_ix  = torch.remainder(left_ix,     rows)  # Wrap around edges of simulation.
  right_ix = torch.remainder(left_ix + 1, rows)
  top_ix   = torch.remainder(top_ix,      cols)
  bot_ix   = torch.remainder(top_ix  + 1, cols)

  # A linearly-weighted sum of the 4 surrounding cells.
  flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
               + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
  return torch.reshape(flat_f, (rows, cols))


def forward(iteration, smoke, vx, vy, output):
  for t in range(1, steps):
    vx_updated = advect(vx, vx, vy)
    vy_updated = advect(vy, vx, vy)
    vx, vy = project(vx_updated, vy_updated)
    smoke = advect(smoke, vx, vy)

    if output:
      matplotlib.image.imsave("output_pytorch/step{0:03d}.png".format(t), 255 * smoke.cpu().detach().numpy())

  return smoke

def main():
  os.system("mkdir -p output_pytorch")
  print("Loading initial and target states...")
  initial_smoke_img = imread("init_smoke.png")[:, :, 0] / 255.0
  target_img = imread("peace.png")[::2, ::2, 3] / 255.0

  vx = torch.zeros(n_grid, n_grid, requires_grad=True, device=device, dtype=torch.float32)
  vy = torch.zeros(n_grid, n_grid, requires_grad=True, device=device, dtype=torch.float32)
  initial_smoke = torch.tensor(initial_smoke_img, device=device, dtype=torch.float32)
  target = torch.tensor(target_img, device=device, dtype=torch.float32)

  for opt in range(num_iterations):
    t = time.time()
    smoke = forward(opt, initial_smoke, vx, vy, opt == (num_iterations - 1))
    loss = ((smoke - target)**2).mean()
    print('forward time', (time.time() - t) * 1000, 'ms')

    t = time.time()
    loss.backward()
    print('backward time', (time.time() - t) * 1000, 'ms')

    with torch.no_grad():
      vx -= learning_rate * vx.grad.data
      vy -= learning_rate * vy.grad.data
      vx.grad.data.zero_()
      vy.grad.data.zero_()

    print('Iter', opt, ' Loss =', loss.item())

if __name__ == '__main__':
  main()


