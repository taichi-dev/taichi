import taichi_lang as ti
import numpy as np
import cv2

res = 512
color_buffer = ti.Vector(3, dt=ti.f32)

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True
grid_resolution = 16

@ti.layout
def buffers():
  ti.root.dense(ti.ij, res).place(color_buffer)

def query_density_int(ipos):
  # return ipos[2] == grid_resolution // 2 and ipos[1] == ipos[0]
  return ipos[0]
  # return 1

@ti.kernel
def render():
  for u, v in color_buffer(0):
    fov = 0.3
    pos = ti.Vector([0.5, 0.5, 2.0])
    d = ti.Vector([fov * u / (res / 2) - fov - 1e-3,
                   fov * v / (res / 2) - fov - 1e-3,
                   -1.0])

    d = ti.Matrix.normalized(d)

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
      last_sample = ipos[0] + ipos[1] + ipos[2] < 10 and ipos.min() >= 0
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

    #for c in ti.static(range(3)):
    #  color_buffer[u, v][c] = i / 500.0
    color_buffer[u, v] = normal * 0.3 + ti.Matrix([0.5, 0.5, 0.5])


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
