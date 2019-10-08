import sys

import sys
import taichi as ti
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import cm
import taichi as tc

real = ti.f32
ti.set_default_fp(real)

max_steps = 4096
vis_interval = 4
output_vis_interval = 16
steps = 204
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

x = vec()
v = vec()

goal = [0.9, 0.15]

n_objects = 1
ground_height = 0.1

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v)
  ti.root.place(loss)
  ti.root.lazy_grad()

dt = 0.002
learning_rate = 1.0

use_toi = False

@ti.kernel
def advance_toi(t: ti.i32):
  for i in range(n_objects):
    old_v = v[t - 1, i]
    new_v = old_v
    old_x = x[t - 1, i]
    new_x = old_x + dt * new_v
    toi = 0.0
    if new_x[1] < ground_height and new_v[1] < 0:
      new_v[1] = -new_v[1]
      toi = -(old_x[1] - ground_height) / old_v[1]
    v[t, i] = new_v
    x[t, i] = x[t - 1, i] + toi * old_v + (dt - toi) * new_v
    
@ti.kernel
def advance_no_toi(t: ti.i32):
  for i in range(n_objects):
    new_v = v[t - 1, i]
    if x[t - 1, i][1] < ground_height and new_v[1] < 0:
      new_v[1] = -new_v[1]
    v[t, i] = new_v
    x[t, i] = x[t - 1, i] + dt * new_v


@ti.kernel
def compute_loss(t: ti.i32):
  # loss[None] = (x[t, 0] - ti.Vector(goal)).norm()
  loss[None] = x[t, 0][1]


gui = tc.core.GUI("Rigid Body", tc.veci(1024, 1024))
canvas = gui.get_canvas()

def forward(output=None, visualize=True):
  interval = vis_interval
  total_steps = steps
  if output:
    interval = output_vis_interval
    # os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
    total_steps *= 2

  for t in range(1, total_steps):
    if use_toi:
      advance_toi(t)
    else:
      advance_no_toi(t)
    
    if (t + 1) % interval == 0 and visualize:
      canvas.clear(0xFFFFFF)
      canvas.circle(tc.vec(x[t, 0][0], x[t, 0][1])).radius(10).color(0x0).finish()
      offset = 0.003
      canvas.path(tc.vec(0.05, ground_height - offset), tc.vec(0.95, ground_height - offset)).radius(2).color(0xAAAAAA).finish()

      if output:
        gui.screenshot('rigid_body/{}/{:04d}.png'.format(output, t))

      gui.update()

  loss[None] = 0
  compute_loss(steps - 1)


def main():
  zoom = len(sys.argv) > 1
  for toi in [False, True]:
    losses = []
    grads = []
    y_offsets = []
    global use_toi
    use_toi = toi
    if zoom:
      ran = np.arange(0.26, 0.3, 0.0001)
    else:
      ran = np.arange(0, 0.3, 0.02)
    for dy in ran:
      y_offsets.append(0.5 + dy)
      x[0, 0] = [0.7, 0.5 + dy]
      v[0, 0] = [-1, -2]
      
      with ti.Tape(loss):
        forward(visualize=False)
        
      print('dy=', dy, 'Loss=', loss[None])
      grads.append(x.grad[0, 0][1])
      losses.append(loss[None])
    
    suffix = ' (Naive)'
    if use_toi:
      suffix = ' (+TOI)'
    if zoom:
      t = '-'
    else:
      t = ':' if use_toi else '.'
    plt.plot(y_offsets, losses, t, label='Loss' + suffix)
    if not zoom:
      plt.plot(y_offsets, grads, label='Gradient' + suffix)

  fig = plt.gcf()
  fig.set_size_inches(4, 3)
  plt.title('The Effect of Time of Impact on Gradients')
  if not zoom:
    # plt.ylim(-6, 2)
    plt.legend(bbox_to_anchor=(0.02, 0.1), loc='lower left', ncol=1)
    # plt.legend(bbox_to_anchor=(1.1, 1.2), ncol=1)
    # plt.legend()
  else:
    plt.legend()
  plt.xlabel('Initial y')
  # plt.tight_layout()
  plt.show()
  plt.show()
  
if __name__ == '__main__':
  main()
