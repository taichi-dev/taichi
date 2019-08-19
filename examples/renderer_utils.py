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
