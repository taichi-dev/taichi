import taichi as ti

ti.init(arch=ti.cpu, kernel_profiler=True, print_ir=True)

quant = True

n = 1024 * 1024 * 256

if quant:
    ci16 = ti.types.quant.int(16, True)

    x = ti.field(dtype=ci16)
    y = ti.field(dtype=ci16)

    ti.root.dense(ti.i, n).bit_struct(num_bits=32).place(x, y)
else:
    x = ti.field(dtype=ti.i16)
    y = ti.field(dtype=ti.i16)

    ti.root.dense(ti.i, n).place(x, y)


@ti.kernel
def foo():
    for i in range(n):
        x[i] = i & 1023
        y[i] = i & 15


for i in range(10):
    foo()

ti.print_kernel_profile_info()
