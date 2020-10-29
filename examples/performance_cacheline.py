import taichi as ti

ti.init(arch=ti.cpu, kernel_profiler=True)

N = 128 * 1024 * 1024

a = ti.field(dtype=ti.i64, shape=N)


@ti.kernel
def fill(stride: ti.template()):
    # Since this kernel is too simple, we use a large block_dim to better amortize the
    # thread pool overhead. By default the scheduler assigns 512 iterations per task.
    ti.block_dim(8192)
    for i in range(N // stride):
        a[i * stride] = i * stride


for stride in [1, 2, 4, 8, 16, 32, 64, 128]:
    ti.kernel_profiler_clear()
    for _ in range(10):
        fill(stride)
    print(f'Stride= {stride}')
    ti.kernel_profiler_print()
    print()
