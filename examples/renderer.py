import taichi_lang as ti
import numpy as np
import cv2
import math

res = 512
color_buffer = ti.Vector(3, dt=ti.f32)

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True
grid_resolution = 16


@ti.layout
def buffers():
  ti.root.dense(ti.ij, res).place(color_buffer)


@ti.func
def query_density_int(ipos):
  return ipos[0] + ipos[1] + ipos[2] < 10 and ipos.min() >= 0


@ti.func
def dda(pos, d):
  rinv = 1.0 / d
  rsign = ti.Vector([0, 0, 0])
  for i in ti.static(range(3)):
    if d[i] > 0:
      rsign[i] = 1
    else:
      rsign[i] = -1

  o = grid_resolution * pos
  ipos = ti.Matrix.floor(o).cast(ti.i32)
  dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
  running = 1
  i = 0
  hit_distance = -1.0
  normal = ti.Vector([0.0, 0.0, 0.0])
  hit_pos = ti.Vector([0.0, 0.0, 0.0])
  while running:
    last_sample = query_density_int(ipos)
    if last_sample:
      mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) - rsign * 0.5) * rinv
      hit_distance = mini.max() * (1 / grid_resolution)
      hit_pos = pos + hit_distance * d
      running = 0
    else:
      mm = ti.Vector([0, 0, 0])
      if dis[0] <= dis[1] and dis[0] < dis[2]:
        mm[0] = 1
      elif dis[1] <= dis[0] and dis[1] <= dis[2]:
        mm[1] = 1
      else:
        mm[2] = 1
      dis += mm * rsign * rinv
      ipos += mm * rsign
      normal = -mm * rsign
    i += 1
    if i > 500:
      running = 0
      normal = [0, 0, 0]
  return normal, hit_pos


@ti.func
def out_dir(n):
  u = ti.Vector([1.0, 0.0, 0.0])
  if ti.abs(n[1]) < 1 - 1e-3:
    u = ti.Matrix.normalized(ti.Matrix.cross(n, ti.Vector([0.0, 1.0, 0.0])))
  v = ti.Matrix.cross(n, u)
  phi = 2 * math.pi * ti.random(ti.f32)
  r = ti.random(ti.f32)
  alpha = 0.5 * math.pi * (r * r)
  return ti.sin(alpha) * (ti.cos(phi) * u + ti.sin(phi) * v) + ti.cos(alpha) * n


@ti.kernel
def render():
  for u, v in color_buffer(0):
    fov = 0.4
    pos = ti.Vector([0.5, 0.5, 3.0])
    d = ti.Vector([fov * u / (res / 2) - fov - 1e-3,
                   fov * v / (res / 2) - fov - 1e-3,
                   -1.0])

    d = ti.Matrix.normalized(d)
    normal, hit_pos = dda(pos, d)

    contrib = ti.Vector([0.0, 0.0, 0.0])


    if normal.norm() != 0:
      contrib += ti.Vector([0.3, 0.3, 0.3])

      d = out_dir(normal)
      normal, _ = dda(hit_pos + 1e-4 * d, d)
      if normal.norm() == 0:
        contrib += ti.Vector([0.3, 0.3, 0.3])

    color_buffer[u, v] = contrib


@ti.kernel
def copy(img: np.ndarray):
  for i, j in color_buffer(0):
    coord = ((res - 1 - j) * res + i) * 3
    for c in ti.static(range(3)):
      img[coord + c] = color_buffer[i, j][c]


def main():
  render()
  img = np.zeros((res * res * 3,), dtype=np.float32)
  copy(img)
  img = img.reshape(res, res, 3)
  cv2.imshow('img', img)
  cv2.waitKey(0)


if __name__ == '__main__':
  main()
