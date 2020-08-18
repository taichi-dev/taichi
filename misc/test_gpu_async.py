import taichi as ti
import time

ti.init(arch=ti.cuda)

a = ti.field(dtype=float, shape=(1024 * 1024 * 1024))


@ti.kernel
def fill(x: float):
    for i in a:
        a[i] = x


for i in range(100):
    t = time.time()
    fill(i)
    print(time.time() - t)

print(a[0])
