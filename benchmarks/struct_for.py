# originally by @KLozes

import taichi as ti
import time
# ti.cfg.arch = ti.x86_64
ti.cfg.arch = ti.cuda
# ti.cfg.print_ir = True

ti.cfg.enable_profiler = True
ti.cfg.verbose_kernel_launches = True

a = ti.var(dt=ti.f32)
N = 512

@ti.layout
def place():
  ti.root.dense(ti.ij, [N,N]).dense(ti.ij, [8,8]).place(a)

@ti.kernel
def set1():
  for i,j in a:
    a[i,j] = 2.0

@ti.kernel
def set2():
  for j in range(N*8):
    for i in range(N*8):
      a[i,j] = 2.0

set1()
set2()

t = time.time()
for n in range(100):
  set1()
elapsed = time.time() - t
ti.get_runtime().sync()
print(elapsed * 10, 'ms/iter')

t = time.time()
for n in range(100):
  set2()
elapsed = time.time() - t
ti.get_runtime().sync()
print(elapsed * 10, 'ms/iter')
ti.profiler_print()
