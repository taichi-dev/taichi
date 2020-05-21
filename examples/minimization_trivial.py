import taichi as ti
import random

n = 8
x = ti.var(dt=ti.f32)
y = ti.var(dt=ti.f32)
L = ti.var(dt=ti.f32)


@ti.layout
def data():
    ti.root.dense(ti.i, n).place(x, y, x.grad,
                                 y.grad)  # place gradient tensors
    ti.root.place(L, L.grad)


@ti.kernel
def reduce():
    global L
    for i in range(n):
        ti.atomic_add(L, 0.5 * (x[i] - y[i])**2)


# Initialize vectors
for i in range(n):
    x[i] = random.random()
    y[i] = random.random()


@ti.kernel
def update():
    for i in x:
        x[i] -= x.grad[i] * 0.1


# Optimize with 100 gradient descent iterations
for k in range(100):
    with ti.Tape(loss=L):
        reduce()
    print('Loss =', L[None])
    update()

for i in range(n):
    # Now you should approximately have x[i] == y[i]
    print(x[i], y[i])
