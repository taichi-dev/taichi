import taichi_lang as ti
import numpy as np
import random
import cv2
import time

real = ti.f32
dim = 2
n_particles = 1024
n_grid = 32
dx = 1.0 / n_grid
dt = 1e-3

def vec():
  return ti.Vector(dim, dt=real)

def mat():
  return ti.Matrix(dim, dim, dt=real)

x, v = vec(), vec()

@ti.layout
def place():
  ti.root.dense(ti.index(2), n_particles).place(x, v)

@ti.kernel
def p2g():
  for i in x(0):
    x[i] = x[i] + v[i] * dt


'''
@ti.kernel
def grid_op():
  pass

@ti.kernel
def g2p():
  pass
'''


def main():
  pass


if __name__ == '__main__':
  for i in range(n_particles):
    x(0)[i] = random.random() * 0.4 + 0.2
    x(1)[i] = random.random() * 0.4 + 0.2
    v(0)[i] = 1
    v(1)[i] = 1
  for f in range(100):
    for s in range(2):
      p2g()
      #grid_op()
      #g2p()
    scale = 10
    img = np.zeros(shape=(scale * n_grid, scale * n_grid))
    for i in range(n_particles):
      p_x = int(scale * x(0)[i] / dx)
      p_y = int(scale * x(1)[i] / dx)
      img[p_x, p_y] = 1
    cv2.imshow('MPM', img)
    cv2.waitKey(1)
  print(x(0)[1])

