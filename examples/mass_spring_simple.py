import taichi_lang as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)

max_steps = 1024
vis_interval = 256
output_vis_interval = 8
steps = 512
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()
x = vec()
v = vec()
force = vec()

n_objects = 3

mass = 1

n_springs = 3
spring_anchor_a = ti.global_var(ti.i32)
spring_anchor_b = ti.global_var(ti.i32)
spring_length = scalar()
spring_stiffness = 10
damping = 10

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, force)
  ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                       spring_length)
  ti.root.place(loss)
  ti.root.lazy_grad()


dt = 0.001
learning_rate = 0.5

@ti.kernel
def apply_spring_force(t: ti.i32):
  # Kernels can have parameters. there t is a parameter with type int32.
  for i in range(n_springs): # A parallel for, preferably on GPUs
    a, b = spring_anchor_a[i], spring_anchor_b[i]
    x_a, x_b = x[t - 1, a], x[t - 1, b]
    dist = x_a - x_b
    length = dist.norm() + 1e-4
    F = (length - spring_length[i]) * spring_stiffness * dist / length
    # apply spring impulses to mass points. Use atomic_add for parallel safety.
    ti.atomic_add(force[t, a],  -F)
    ti.atomic_add(force[t, b],  F)

@ti.kernel
def time_integrate(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * damping)
    v[t, i] = s * v[t - 1, i] + dt * force[t, i] / mass
    x[t, i] = x[t - 1, i] + dt * v[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
  x01 = x[t, 0] - x[t, 1]
  x02 = x[t, 0] - x[t, 2]
  area = ti.abs(0.5 * (x01[0] * x02[1] - x01[1] * x02[0])) # area from cross product
  target_area = 0.1
  loss[None] = ti.sqr(area - target_area)


def visualize(output, t):
  img = np.ones(shape=(vis_resolution, vis_resolution, 3),
                dtype=np.float32) * 0.8
  
  def circle(x, y, color):
    radius = 0.02
    cv2.circle(img, center=(
      int(vis_resolution * x), int(vis_resolution * (1 - y))),
               radius=int(radius * vis_resolution), color=color,
               thickness=-1)
  
  for i in range(n_objects):
    color = (0.4, 0.6, 0.6)
    circle(x[t, i][0], x[t, i][1], color)
  
  for i in range(n_springs):
    def get_pt(x):
      return int(x[0] * vis_resolution), int(vis_resolution - x[1]* vis_resolution)
    act = 0
    cv2.line(img, get_pt(x[t, spring_anchor_a[i]]), get_pt(x[t, spring_anchor_b[i]]), (0.5 + act, 0.5, 0.5 - act), thickness=6)
  
  cv2.imshow('img', img)
  cv2.waitKey(1)
  if output:
    cv2.imwrite('mass_spring/{}/{:04d}.png'.format(output, t), img * 255)

def forward(output=None):
  interval = vis_interval
  if output:
    interval = output_vis_interval
    os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)
    
  total_steps = steps if not output else steps * 2
  
  for t in range(1, total_steps):
    apply_spring_force(t)
    time_integrate(t)
    
    if (t + 1) % interval == 0:
      visualize(output, t)
  
  loss[None] = 0
  compute_loss(steps - 1)


@ti.kernel
def clear_states():
  for t in range(0, max_steps):
    for i in range(0, n_objects):
      x.grad[t, i] = ti.Vector([0.0, 0.0])
      v.grad[t, i] = ti.Vector([0.0, 0.0])
      force[t, i] = ti.Vector([0.0, 0.0])
      force.grad[t, i] = ti.Vector([0.0, 0.0])

@ti.kernel
def clear_springs():
  for i in range(n_springs):
    spring_length.grad[i] = 0.0

def clear():
  clear_states()
  clear_springs()

def main():
  x[0, 0] = [0.3, 0.3]
  x[0, 1] = [0.3, 0.4]
  x[0, 2] = [0.4, 0.4]
  
  spring_anchor_a[0], spring_anchor_b[0], spring_length[0] = 0, 1, 0.1
  spring_anchor_a[1], spring_anchor_b[1], spring_length[1] = 1, 2, 0.1
  spring_anchor_a[2], spring_anchor_b[2], spring_length[2] = 2, 0, 0.1
  
  losses = []
  for iter in range(1000):
    clear()
    
    with ti.Tape(loss):
      forward()
    
    print('Iter=', iter, 'Loss=', loss[None])
    losses.append(loss[None])

    for i in range(n_springs):
      print(spring_length.grad[i])
    for i in range(n_springs):
      spring_length[i] -= learning_rate * spring_length.grad[i]
  
  plt.plot(losses)
  plt.show()
  
  clear()
  forward('final')


if __name__ == '__main__':
  main()