import taichi as ti
import time

ti.init(arch=ti.cuda)

a = ti.var(dt=ti.f32, shape=(1024 * 1024 * 1024))


@ti.kernel
def fill(x: ti.f32):
    for i in a:
        a[i] = x


for i in range(100):
    t = time.time()
    fill(i)
    print(time.time() - t)

print(a[0])
