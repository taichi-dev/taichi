import taichi as ti

ti.init(ti.opengl, log_level=ti.DEBUG, saturating_grid_dim=1)

N = 1024
a = ti.field(int, ())
x = ti.field(int, N)

@ti.kernel
def func():
    for i in range(N):
        x[i] = i + 1

a[None] = 1023
func()
print(x)
