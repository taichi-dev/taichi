import taichi as ti
import timeit

ti.init(ti.opengl, log_level=ti.DEBUG)

N = 2**14

a = ti.field(int, 4)
x = ti.field(int, N)

@ti.kernel
def indirect():
    for i in range(a[3]):
        x[i] = i + 1

a[0] = 128
a[1] = 1
a[2] = 1
a[3] = N - 1
stmt = lambda: indirect()
print(timeit.timeit(stmt, stmt, number=10000))
print(x)
print(a)
