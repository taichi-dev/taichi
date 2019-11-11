import taichi as ti

print(1)
x, y = ti.var(ti.f32), ti.var(ti.f32)
print(2)

@ti.layout
def xy():
  print(3)
  ti.root.dense(ti.ij, 16).place(x, y)

print(4)

@ti.kernel
def laplace():
  print(5)
  for i, j in x:
    if (i + j) % 3 == 0:
      y[i, j] = 4.0 * x[i, j] - x[i - 1, j] - x[i + 1, j] - x[i, j - 1] - x[i, j + 1]
    else:
      y[i, j] = 0.0

exit()

print(6)
for i in range(10):
  x[i, i + 1] = 1.0

print(7)
laplace()
print(8)

for i in range(10):
  print(y[i, i + 1])
