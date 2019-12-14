import taichi as ti

@ti.all_archs
def test_random():
  n = 1024
  x = ti.var(ti.f32, shape=n)

  @ti.kernel
  def fill():
    for i in range(n):
      x[i] = ti.random()
      
  fill()
  for i in range(n):
    print(x[i])

