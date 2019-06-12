import taichi_lang as ti
import taichi as tc
import matplotlib.pyplot as plt
import random

tc.set_gdb_trigger(True)

number_coeffs = 3

N = 128
x, y = ti.var(ti.f32), ti.var(ti.f32)
coeffs = [ti.var(ti.f32)] * number_coeffs
loss = ti.var(ti.f32)

@ti.layout
def xy():
  ti.root.dense(ti.i, N).place(x, x.grad, y, y.grad)
  ti.root.place(loss, loss.grad)
  for i in range(number_coeffs):
    ti.root.place(coeffs[i], coeffs[i].grad)

ti.cfg.print_ir = True
@ti.kernel
def regress():
  for i in x:
    v = x[i]
    est = 0.0
    for i in ti.static(range(number_coeffs)):
      est += coeffs[i] * ti.pow(v, i)
    loss.atomic_add(0.5 * ti.sqr(y[i] - est))

xs = []
ys = []
grad_xs = []

for i in range(N):
  v = random.random() * 5 - 2.5
  xs.append(v)
  x[i] = v
  y[i] = (v - 1) * (v - 2) * (v + 2) + random.random() - 0.5

regress()

print('y')
for i in range(N):
  y.grad[i] = 1
  ys.append(y[i])
print()

for i in range(10):
  regress.grad()
  print('grad_x')
  for i in range(N):
    grad_xs.append(x.grad[i])

plt.title('Nonlinear Regression')
ax = plt.gca()
ax.scatter(xs, ys, label='f(x)')
ax.legend()
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.show()

