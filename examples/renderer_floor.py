import taichi as ti
import os
import numpy as np
import math
import time
import random
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi
import sys

res = 1280, 720
num_spheres = 1024
color_buffer = ti.Vector(3, dt=ti.f32)
max_ray_depth = 1
use_directional_light = True

fov = 0.23
dist_limit = 100

exposure = 1.5
camera_pos = ti.Vector([0.5, 0.32, 2.7])
light_direction = [1.2, 0.3, 0.7]
light_direction_noise = 0.03
light_color = [1.0, 1.0, 1.0]

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True
ti.cfg.arch = ti.cuda

camera_pos = ti.Vector([0.5, 0.27, 2.7])

@ti.layout
def buffers():
  ti.root.dense(ti.ij, (res[0] // 8, res[1] // 8)).dense(ti.ij, 8).place(color_buffer)


@ti.func
def sdf(o):
  return o[1] - 0.027


@ti.func
def ray_march(p, d):
  j = 0
  dist = 0.0
  limit = 200
  while j < limit and sdf(p + dist * d) > 1e-8 and dist < dist_limit:
    dist += sdf(p + dist * d)
    j += 1
  if dist > dist_limit:
    dist = inf
  return dist


@ti.func
def sdf_normal(p):
  d = 1e-3
  n = ti.Vector([0.0, 0.0, 0.0])
  for i in ti.static(range(3)):
    inc = p
    dec = p
    inc[i] += d
    dec[i] -= d
    n[i] = (0.5 / d) * (sdf(inc) - sdf(dec))
  return ti.Matrix.normalized(n)


@ti.func
def next_hit(pos, d):
  closest = inf
  normal = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])

  ray_march_dist = ray_march(pos, d)
  if ray_march_dist < dist_limit and ray_march_dist < closest:
    closest = ray_march_dist
    normal = sdf_normal(pos + d * closest)
    c = [1, 0.5, 0.5]

  return closest, normal, c


aspect_ratio = res[0] / res[1]

@ti.kernel
def render():
  for u, v in color_buffer:
    pos = camera_pos
    
    contrib = ti.Vector([0.0, 0.0, 0.0])
    closest, normal, c = next_hit(pos, ti.Vector([0.0, 0.0, -1.0]))
    n = normal.norm()
    contrib = [n, n, n]

    color_buffer[u, v] = contrib


@ti.kernel
def copy(img: ti.ext_arr()):
  for i, j in color_buffer:
    for c in ti.static(range(3)):
      img[i, j, c] = color_buffer[i, j][c]


def main():
  gui = ti.GUI('Particle Renderer', res)

  render()
  while True:
    gui.set_image(color_buffer.to_numpy(as_vector=True))
    gui.show()


if __name__ == '__main__':
  main()
