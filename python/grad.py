import taichi_lang as ti
import taichi as tc
import matplotlib.pyplot as plt

tc.set_gdb_trigger(True)

N = 2048
x, y = ti.var(ti.f32), ti.var(ti.f32)

ti.cfg.lower_access = False

@ti.layout
def xy():
  ti.root.dense(ti.i, N).place(x, y, x.grad, y.grad)

@ti.kernel
def poly():
  for i in x:
    v = x[i]
    ret = 0.0
    if v < -1 or v > 1:
      ret = v - 1
    else:
      ret = v * v * 0.5 - 3
    y[i] = ti.sin(5 * ret)

xs = []
ys = []
grad_xs = []

for i in range(N):
  v = ((i + 0.5) / N) * 7 - 3
  xs.append(v)
  x[i] = v

poly()

print('y')
for i in range(N):
  y.grad[i] = 1
  ys.append(y[i])
print()

poly.grad()
print('grad_x')
for i in range(N):
  grad_xs.append(x.grad[i])

plt.title('Auto Diff')
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

