import taichi_lang as ti

x, y = ti.var(ti.f32), ti.var(ti.f32)

@ti.layout
def place_variables():
  ti.root.dense(ti.indices(0, 1), (16, 16)).place(x.ptr).place(y.ptr)

@ti.kernel
def laplace():
  for i, j in x:
    y[i, j] = 4.0 * x[i, j] - x[i - 1, j] - x[i + 1, j] - x[i, j - 1] - x[i, j + 1]

for i in range(10):
  x[i, i + 1] = 1.0

laplace()
for i in range(10):
  print(y[i, i + 1])
