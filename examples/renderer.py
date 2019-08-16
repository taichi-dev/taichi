import taichi_lang as ti
import numpy as np
import cv2

res = 1024
color_buffer = ti.Vector(3, dt=ti.f32)

ti.runtime.print_preprocessed = True


@ti.layout
def buffers():
  ti.root.dense(ti.ij, res).place(color_buffer)



@ti.kernel
def render():
  for i, j in color_buffer(0):
    fov = 1
    orig = ti.Vector([0.0, 0.0, 12.0])
    c = ti.Vector([fov * i / (res / 2) - 1.0,
                   fov * j / (res / 2) - 1.0,
                   -1.0])

    c = ti.Matrix.normalized(c)

    hit = 0

    for k in range(300):
      p = orig + k * 0.1 * c
      if ti.Matrix.norm(p) < 2:
        hit = 1

    color_buffer[i, j] = c * hit


@ti.kernel
def copy(img: np.ndarray):
  for i, j in color_buffer(0):
    coord = (i * res + j) * 3
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
