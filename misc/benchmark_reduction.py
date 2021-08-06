import taichi as ti

# TODO: make this a real benchmark and set up regression

ti.init(arch=ti.gpu)

N = 1024 * 1024 * 1024

a = ti.field(ti.i32, shape=N)
tot = ti.field(ti.i32, shape=())


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
ti.print_kernel_profile_info()
