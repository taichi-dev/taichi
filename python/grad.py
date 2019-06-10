import taichi_lang as ti

N = 8
x, y = ti.var(ti.f32), ti.var(ti.f32)
grad_x, grad_y = ti.var(ti.f32), ti.var(ti.f32)

@ti.layout
def xy():
  ti.root.dense(ti.i, N).place(x, y, grad_x, grad_y)

@ti.kernel
def poly():
  for i in x:
    y[i] = x[i] + x[i]

for i in range(N):
  x[i] = i / N

poly()

for i in range(N):
  print(y[i])
