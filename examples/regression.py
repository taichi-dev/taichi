import taichi as ti
import taichi as tc
import matplotlib.pyplot as plt
import random
import numpy as np

tc.set_gdb_trigger(True)

number_coeffs = 4
learning_rate = 1e-4

N = 32
x, y = ti.var(ti.f32), ti.var(ti.f32)
coeffs = [ti.var(ti.f32) for _ in range(number_coeffs)]
loss = ti.var(ti.f32)


@ti.layout
def xy():
    ti.root.dense(ti.i, N).place(x, x.grad, y, y.grad)
    ti.root.place(loss, loss.grad)
    for i in range(number_coeffs):
        ti.root.place(coeffs[i], coeffs[i].grad)


@ti.kernel
def regress():
    for i in x:
        v = x[i]
        est = 0.0
        for j in ti.static(range(number_coeffs)):
            est += coeffs[j] * (v**j)
        loss[None] += 0.5 * (y[i] - est)**2


@ti.kernel
def update():
    for i in ti.static(range(number_coeffs)):
        coeffs[i][None] -= learning_rate * coeffs[i].grad[None]
        coeffs[i].grad[None] = 0


xs = []
ys = []

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

for i in range(1000):
    loss[None] = 0
    loss.grad[None] = 1
    regress()
    regress.grad()
    print('Loss =', loss[None])
    update()
    for i in range(number_coeffs):
        print(coeffs[i][None], end=', ')
    print()

curve_xs = np.arange(-2.5, 2.5, 0.01)
curve_ys = curve_xs * 0
for i in range(number_coeffs):
    curve_ys += coeffs[i][None] * np.power(curve_xs, i)

plt.title('Nonlinear Regression with Gradient Descent (3rd order polynomial)')
ax = plt.gca()
ax.scatter(xs, ys, label='data', color='r')
ax.plot(curve_xs, curve_ys, label='fitted')
ax.legend()
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.show()
