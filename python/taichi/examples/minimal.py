import taichi as ti

ti.init(arch=ti.cpu, print_ir=True)

arr_ = ti.ndarray(dtype=ti.math.mat2, shape=(8, 8))
arr_[0, 0] = ti.math.mat2([[1, 2], [3, 4]])


@ti.kernel
def p(arr: ti.types.ndarray(dtype=ti.math.mat2, ndim=2)) -> ti.math.mat2:
    return arr[0, 0]


print(p(arr_))
