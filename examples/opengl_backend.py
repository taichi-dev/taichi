import taichi as ti

ti.init(arch=ti.opengl)

x = ti.var(ti.f32, shape=(2, 4))

@ti.kernel
def func():
  x[1, 2] = 666
  x[2, 1] = 527

x[1, 2] = 233
x[1, 3] = 888
func()
print(x[1, 2])
print(x[1, 3])
print(x[2, 1])
