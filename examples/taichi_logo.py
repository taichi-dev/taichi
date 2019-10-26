# TODO: this program crashes clang-7 when compiled with ti.x86_64. Why?
import taichi as ti
import cv2
import numpy as np

def Vector2(x, y):
  return ti.Vector([x, y])

@ti.func
def inside(p, c, r):
  return (p - c).norm_sqr() <= r * r

@ti.func
def inside_taichi(p_):
  p = p_
  p = Vector2(0.5, 0.5) + (p - Vector2(0.5, 0.5)) * 1.11
  ret = -1
  if not inside(p, Vector2(0.50, 0.50), 0.55):
    if ret == -1:
      ret = 0
  if not inside(p, Vector2(0.50, 0.50), 0.50):
    if ret == -1:
      ret = 1
  if inside(p, Vector2(0.50, 0.25), 0.09):
    if ret == -1:
      ret = 1
  if inside(p, Vector2(0.50, 0.75), 0.09):
    if ret == -1:
      ret = 0
  if inside(p, Vector2(0.50, 0.25), 0.25):
    if ret == -1:
      ret = 0
  if inside(p, Vector2(0.50, 0.75), 0.25):
    if ret == -1:
      ret = 1
  if p[0] < 0.5:
    if ret == -1:
      ret = 1
  else:
    if ret == -1:
      ret = 0
  return ret

x = ti.var(ti.f32)

n = 512
ti.cfg.use_llvm = True

@ti.layout
def layout():
  ti.root.dense(ti.ij, n).place(x)


@ti.kernel
def paint():
  for i in range(n * 4):
    for j in range(n * 4):
      ret = 1.0 - inside_taichi(Vector2(1.0 * i / n / 4, 1.0 * j / n / 4))
      x[i // 4, j // 4] += ret / 16


paint()

img = np.empty((n, n), dtype=np.float32)
for i in range(n):
  for j in range(n):
    img[i, j] = x[j, n - 1 - i]

cv2.imshow('Taichi', img)
cv2.imwrite('taichi.png', img * 255)
cv2.waitKey(0)
