import taichi as ti

ti.init(arch=ti.opengl)

x = ti.var(ti.i32, shape=(3))

@ti.kernel
def func(t: ti.i32):
  x[0] = 233 + t
  x[1] = 666

print(x[0])
func(123)
print(x[0])
