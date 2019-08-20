import taichi_lang as ti
import numpy as np
import math

@ti.kernel
def copy(img: np.ndarray):
  for i, j in color_buffer(0):
    coord = ((res - 1 - j) * res + i) * 3
    for c in ti.static(range(3)):
      img[coord + c] = color_buffer[i, j][2 - c]

@ti.func
def out_dir(n):
  u = ti.Vector([1.0, 0.0, 0.0])
  if ti.abs(n[1]) < 1 - 1e-3:
    u = ti.Matrix.normalized(ti.Matrix.cross(n, ti.Vector([0.0, 1.0, 0.0])))
  v = ti.Matrix.cross(n, u)
  phi = 2 * math.pi * ti.random(ti.f32)
  r = ti.random(ti.f32)
  ay = ti.sqrt(r)
  ax = ti.sqrt(1 - r)
  return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n

@ti.func
def ray_aabb_intersection(box_min, box_max, o, d):
  intersect = 1

  near_int = -1e10
  far_int = 1e10

  for i in ti.static(range(3)):
    if d[i] == 0:
      if o[i] < box_min[i] or o[i] > box_max[i]:
        intersect = 0
    else:
      i1 = (box_min[i] - o[i]) / d[i]
      i2 = (box_max[i] - o[i]) / d[i]

      new_far_int = ti.max(i1, i2)
      new_near_int = ti.min(i1, i2)

      far_int = ti.min(new_far_int, far_int)
      near_int = ti.max(new_near_int, near_int)


  if near_int > far_int:
    intersect = 0
  return intersect, near_int, far_int

