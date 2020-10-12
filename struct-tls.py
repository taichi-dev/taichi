import taichi as ti

ti.init(arch=ti.cuda, print_ir=True, print_kernel_llvm_ir=False)

n = 1024
x = ti.field(dtype=ti.i32)
ti.root.pointer(ti.i, n // 32).dense(ti.i, 32).place(x)

for i in range(n):
    x[i] = i


@ti.kernel
def reduce() -> ti.i32:
    sum = 0
    for i in x:
        sum += x[i]
    return sum


ret = reduce()
print(ret)
assert ret == (n * (n - 1)) // 2
