import taichi as ti

ti.init(arch=ti.cpu, print_ir=True, print_kernel_llvm_ir=False, kernel_profiler=True, make_thread_local=False)

dense = False

n = 1024 * 1024 * 128
x = ti.field(dtype=ti.i32)
if dense:
    ti.root.dense(ti.i, n // 512).dense(ti.i, 512).place(x)
else:
    ti.root.pointer(ti.i, n // 512).dense(ti.i, 512).place(x)

@ti.kernel
def fill():
    for i in range(n):
        x[i] = i

fill()

@ti.kernel
def reduce() -> ti.i32:
    sum = 0
    for i in x:
        x[i] += 1
    return sum


for i in range(100):
    ret = reduce()

print(ret)
ti.kernel_profiler_print()
