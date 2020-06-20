import taichi as ti

ti.init(print_ir=True, kernel_profiler=True)
# ti.init(kernel_profiler=True)
# ti.core.toggle_advanced_optimization(False)

N = 1024 * 1024

a = ti.var(ti.i32, shape=N)
tot = ti.var(ti.i32, shape=())

@ti.kernel
def fill():
    for i in a:
        a[i] = 3

@ti.kernel
def reduce():
    for i in a:
        tot[None] += a[i]
        
fill()

for i in range(10):
    reduce()

print(tot[None])
assert tot[None] == 3 * 10 * N
ti.kernel_profiler_print()
