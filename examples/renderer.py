import taichi_lang as ti
import numpy as np
import cv2

res = 1024
color_buffer = ti.Vector(3, dt=ti.f32)

@ti.layout
def buffers():
  ti.root.dense(ti.ij, res).place(color_buffer)

@ti.kernel
def render():
  for i, j in color_buffer(0):
    color_buffer[i, j](0).val = i
    color_buffer[i, j](1).val = j
    color_buffer[i, j](2).val = i + j

@ti.kernel
def copy(img: np.ndarray):
  for i, j in color_buffer(0):
    coord = (i * res + j) * 3
    for c in ti.static(range(3)):
      img[coord + c] = color_buffer[i, j](c) * 0.003

def main():
  render()
  img = np.zeros((res * res * 3, ), dtype=np.float32)
  copy(img)
  img = img.reshape(res, res, 3)
  cv2.imshow('img', img)
  cv2.waitKey(0)

if __name__ == '__main__':
  main()