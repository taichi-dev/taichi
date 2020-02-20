import taichi as ti

ti.init(arch=ti.opengl)

x = ti.var(ti.f64, shape=(2))

@ti.kernel
def func():
  x[0] = 666

x[0] = 233
func()
print(x[0])
