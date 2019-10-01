import taichi as ti


x = ti.Matrix(2, 2, ti.i32)


@ti.layout
def xy():
  ti.root.dense(ti.i, 16).place(x)


@ti.kernel
def inc():
  for i in x(0, 0):
    delta = ti.Matrix(2, 2)
    delta(1, 1).assign(3)
    x(1, 1)[i] = x(0, 0)[i] + 1
    x[i] = x[i] + delta


for i in range(10):
  x(0, 0)[i] = i

inc()

for i in range(10):
  print(x(1, 1)[i])
