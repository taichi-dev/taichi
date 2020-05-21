import taichi as ti
import time

ti.init(async=True)

x = ti.var(ti.i32)
y = ti.var(ti.i32)
z = ti.var(ti.i32)

ti.root.dense(ti.i, 1024**3).place(x)
ti.root.dense(ti.i, 1024**3).place(y)
ti.root.dense(ti.i, 1024**3).place(z)


@ti.kernel
def x_to_y():
    for i in x:
        y[i] = x[i] + 1


@ti.kernel
def y_to_z():
    for i in x:
        z[i] = y[i] + 4


@ti.kernel
def inc():
    for i in x:
        x[i] = x[i] + 1


n = 100

for i in range(n):
    x[i] = i * 10

repeat = 10

for i in range(repeat):
    t = time.time()
    x_to_y()
    ti.sync()
    print('x_to_y', time.time() - t)

for i in range(repeat):
    t = time.time()
    y_to_z()
    ti.sync()
    print('y_to_z', time.time() - t)

for i in range(repeat):
    t = time.time()
    x_to_y()
    y_to_z()
    ti.sync()
    print('fused x->y->z', time.time() - t)

for i in range(repeat):
    t = time.time()
    inc()
    ti.sync()
    print('single inc', time.time() - t)

for i in range(repeat):
    t = time.time()
    for j in range(10):
        inc()
    ti.sync()
    print('fused 10 inc', time.time() - t)

for i in range(n):
    assert x[i] == i * 10
    assert y[i] == x[i] + 1
    assert z[i] == x[i] + 5
