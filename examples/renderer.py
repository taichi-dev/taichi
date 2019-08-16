import taichi_lang as ti
import numpy as np
import cv2

res = 128
color_buffer = ti.Vector(3, dt=ti.f32)

@ti.layout
def buffers():
  ti.root.dense(ti.ij, res).place(color_buffer)

@ti.kernel
def render():
  for i, j in color_buffer(0):
    color_buffer[i, j](0).val = i + j

@ti.kernel
def copy(img: np.ndarray):
  for i, j in color_buffer(0):
    img[i * res + j] = color_buffer[i, j](0) * 0.01

def main():
  render()
  img = np.zeros((res * res, ), dtype=np.float32)
  copy(img)
  img = img.reshape(res, res, 1)
  while True:
    cv2.imshow('img', img)
    cv2.waitKey(1)

if __name__ == '__main__':
  main()