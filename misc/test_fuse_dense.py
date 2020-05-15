import taichi as ti
import time

ti.init()

x = ti.var(ti.i32)
y = ti.var(ti.i32)
z = ti.var(ti.i32)

ti.root.dense(ti.i, 1024 ** 3).place(x, y, z)

@ti.kernel
def x_to_y():
    for i in x:
        y[i] = x[i] + 1
        
@ti.kernel
def y_to_z():
    for i in x:
        z[i] = y[i] + 4
        

n = 1024

for i in range(n):
   x[i] = i * 10
   
x_to_y()
ti.sync()

for i in range(10):
    t = time.time()
    x_to_y()
    ti.sync()
    print(time.time() - t)

for i in range(10):
    t = time.time()
    y_to_z()
    ti.sync()
    print(time.time() - t)
    
for i in range(10):
    t = time.time()
    x_to_y()
    y_to_z()
    ti.sync()
    print('fused', time.time() - t)


for i in range(n):
    x[i] = i * 10
    assert y[i] == x[i] + 1
    assert z[i] == x[i] + 5

