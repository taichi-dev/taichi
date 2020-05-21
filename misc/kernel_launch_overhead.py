import taichi as ti
import time

ti.init()


@ti.kernel
def compute_div(a: ti.i32):
    pass


compute_div(0)
print("starting...")
t = time.time()
for i in range(100000):
    compute_div(0)
print((time.time() - t) * 10, 'us')
exit(0)
