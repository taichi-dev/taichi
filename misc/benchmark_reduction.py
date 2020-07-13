import taichi as ti

# TODO: make this a real benchmark and set up regression

ti.init(arch=ti.gpu,
        print_ir=True,
        print_kernel_llvm_ir=True,
        kernel_profiler=True,
        print_kernel_llvm_ir_optimized=True)
# ti.init(kernel_profiler=True)
# ti.core.toggle_advanced_optimization(False)

N = 1024 * 1024 * 1024

a = ti.var(ti.i32, shape=N)
tot = ti.var(ti.i32, shape=())


@ti.kernel
def fill():
    ti.block_dim(128)
    for i in a:
        a[i] = i


@ti.kernel
def reduce():
    ti.block_dim(1024)
    for i in a:
        tot[None] += a[i]


fill()
fill()

for i in range(10):
    reduce()

ground_truth = 10 * N * (N - 1) / 2 % 2**32
assert tot[None] % 2**32 == ground_truth
ti.kernel_profiler_print()
