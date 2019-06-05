import taichi_lang as ti
import time

x, y = ti.var(ti.f32), ti.var(ti.f32)

@ti.kernel
def laplace():
  for i, j in x:
    y[i, j] = 4.0 * x[i, j] - x[i - 1, j] - x[i + 1, j] - x[i, j - 1] - x[i, j + 1]

@ti.layout
def place_variables():
  ti.root.dense(ti.indices(0, 1), (16, 16)).place(x.ptr).place(y.ptr)

laplace()

t = time.time()
N = 1000000
for i in range(N):
  x[i & 7, i & 7] = 1.0
print((time.time() - t) / N * 1e9, 'ns')


t = time.time()
N = 1000000
a = 0
for i in range(N):
  a += x[i, i]
print((time.time() - t) / N * 1e9, 'ns')


