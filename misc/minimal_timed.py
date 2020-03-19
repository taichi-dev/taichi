import taichi as ti
import time

t = time.time()
ti.init(arch=ti.cuda)


@ti.kernel
def p():
    print(42)


p()

print(f'{time.time() - t:.3f} s')
