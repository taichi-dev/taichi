import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True, print_ir=True, print_kernel_llvm_ir_optimized=True)
x, y = ti.var(ti.f32), ti.var(ti.f32)

N = 16
bs = 16

# block = ti.root.pointer(ti.ij, N // bs)
# block.dense(ti.ij, bs).place(x)
# block.dense(ti.ij, bs).place(y)
ti.root.pointer(ti.ij, N // bs).dense(ti.ij, bs).place(x, y)


@ti.kernel
def populate():
    for i,j in ti.ndrange(N, N):
        x[i, j] = i - j

@ti.kernel
def copy():
    ti.cache_shared(x)
    for i, j in x:
        print(i, j, x[i, j])
        y[i, j] = x[i, j]


populate()
copy()

for i in range(N):
    for j in range(N):
        print(i, j, y[i, j])
        assert y[i, j] == i - j

ti.kernel_profiler_print()
