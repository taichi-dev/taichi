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
    color_buffer[0][i, j] = i
    color_buffer[1][i, j] = j
    color_buffer[2][i, j] = i + j

@ti.kernel
def copy(img: np.ndarray):
  for i, j in color_buffer(0):
    coord = (i * res + j) * 3
    for c in ti.static(range(3)):
      img[coord + c] = color_buffer[c][i, j] * 0.003

def main():
  render()
  img = np.zeros((res * res * 3, ), dtype=np.float32)
  copy(img)
  img = img.reshape(res, res, 3)
  cv2.imshow('img', img)
  cv2.waitKey(0)

if __name__ == '__main__':
  main()