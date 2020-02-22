import taichi as ti

ti.init(arch=ti.opengl)

x = ti.var(ti.i32, shape=(4, 4))

@ti.kernel
def func():
  for i, j in x:
    x[i, j] = 200 + 10 * i + j

func()
print(x.to_numpy())
