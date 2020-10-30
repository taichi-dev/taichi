import taichi as ti
import timeit

ti.init(ti.opengl, log_level=ti.DEBUG)

N = 2**14

a = ti.field(int, ())
x = ti.field(int, N)

@ti.kernel
def indirect():
    for i in range(a[None]):
        x[i] = i + 1

a[None] = N - 1
stmt = lambda: indirect()
print(timeit.timeit(stmt, stmt, number=10000))
print(x)
print(a)
