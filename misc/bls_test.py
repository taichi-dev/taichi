import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True, print_ir=True, print_kernel_llvm_ir_optimized=True)
x, y = ti.var(ti.f32), ti.var(ti.f32)

N = 48
bs = 16

# block = ti.root.pointer(ti.ij, N // bs)
# block.dense(ti.ij, bs).place(x)
# block.dense(ti.ij, bs).place(y)
ti.root.pointer(ti.i, N // bs).dense(ti.i, bs).place(x, y)


@ti.kernel
def populate():
    for i in range(bs, N - bs):
        x[i] = i

@ti.kernel
def copy():
    ti.cache_shared(x)
    for i in x:
        print(x[i])
        # y[i] = x[i]
        # print(y[i])


populate()
for i in range(10):
    copy()

ti.kernel_profiler_print()
