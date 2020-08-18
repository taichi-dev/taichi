import taichi as ti
import time

ti.init()

c = ti.field(float, ())


#@ti.kernel
def set_c(x: float):
    c[None] = x


set_c(1)
print("starting...")
t = time.time()
for i in range(100000):
    set_c(1)
print((time.time() - t) * 10, 'us')
