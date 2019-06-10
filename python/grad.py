import taichi_lang as ti
import taichi as tc
import matplotlib.pyplot as plt

tc.set_gdb_trigger(True)

N = 128
x, y = ti.var(ti.f32), ti.var(ti.f32)
grad_x, grad_y = ti.var(ti.f32), ti.var(ti.f32)

ti.cfg.lower_access = False

@ti.layout
def xy():
  ti.root.dense(ti.i, N).place(x, y, grad_x, grad_y)
  x.set_grad(grad_x)
  y.set_grad(grad_y)

@ti.kernel
def poly():
  for i in x:
    v = x[i]
    v += 1
    # if v > 2:
    #  v = 0
    y[i] = (v - 1) * (v + 2) * (v - 3)

xs = []
ys = []
grad_xs = []

for i in range(N):
  v = (i / (N - 1)) * 7 - 3
  xs.append(v)
  x[i] = v

poly()

print('y')
for i in range(N):
  print(y[i])
  ys.append(y[i])
  grad_y[i] = 1
print()

poly.grad()
print('grad_x')
for i in range(N):
  grad_xs.append(grad_x[i])
  print(grad_x[i])

plt.title('Auto Diff (x - 1)(x + 2)(x - 3)')
ax = plt.gca()
ax.plot(xs, ys, label='f(x)')
ax.plot(xs, grad_xs, label='f\'(x)')
ax.legend()
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.show()

