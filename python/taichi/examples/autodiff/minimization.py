import random

import taichi as ti

ti.init(arch=ti.cpu)

n = 8
x = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
y = ti.field(dtype=ti.f32, shape=n)
L = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def reduce():
    for i in range(n):
        L[None] += 0.5 * (x[i] - y[i])**2


# Initialize vectors
for i in range(n):
    x[i] = random.random()
    y[i] = random.random()


@ti.kernel
def gradient_descent():
    for i in x:
        x[i] -= x.grad[i] * 0.1


# Optimize with 100 gradient descent iterations
for k in range(100):
    with ti.Tape(loss=L):
        reduce()
    print('Loss =', L[None])
    gradient_descent()

for i in range(n):
    # Now you should approximately have x[i] == y[i]
    print(x[i], y[i])
