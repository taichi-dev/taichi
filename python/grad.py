import taichi_lang as ti
import taichi as tc
import matplotlib.pyplot as plt

tc.set_gdb_trigger(True)

N = 2048
x, y = ti.var(ti.f32), ti.var(ti.f32)

@ti.layout
def xy():
  ti.root.dense(ti.i, N).place(x, x.grad, y, y.grad)

#ti.cfg.lower_access = True
#ti.cfg.print_ir = True

@ti.kernel
def poly():
  for i in x:
    v = x[i]
    ret = 0.0
    guard = 0.2
    if v < -guard or v > guard:
      ret = ti.sin(4 / v)
    else:
      ret = 0
    y[i] = ret

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
  # print(x[i], x.grad[i], y[i], y.grad[i])
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

