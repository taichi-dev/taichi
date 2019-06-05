import taichi_lang as ti

x, y = ti.var(ti.f32), ti.var(ti.f32)

@ti.layout
def xy():
  ti.root.dense(ti.ij, 16).place(x, y)

@ti.kernel
def laplace():
  for i, j in x:
    y[i, j] = 4.0 * x[i, j] - x[i - 1, j] - x[i + 1, j] - x[i, j - 1] - x[i, j + 1]

for i in range(10):
  x[i, i + 1] = 1.0

laplace()
for i in range(10):
  print(y[i, i + 1])
