import taichi as ti
import timeit

ti.init(ti.opengl)

N = 2**14

x = ti.field(int, N)

@ti.kernel
def func():
    for i in x:
        x[i] = i
    for i in range(x[2]):
        x[i] = i

stmt = lambda: func()
print(timeit.timeit(stmt, stmt, number=10000))
