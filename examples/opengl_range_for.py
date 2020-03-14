import taichi as ti

ti.init(arch=ti.opengl)

x = ti.var(ti.i32, shape=(5, 5))

@ti.kernel
def func():
  for i in range(5):
    for j in range(5):
      x[i, j] = i + j

func()
print(x[2, 3])
