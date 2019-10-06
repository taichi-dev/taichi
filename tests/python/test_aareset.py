import taichi as ti

def test_abs():
  for i in range(3):
    ti.reset()
    y = ti.var(ti.f32)

    @ti.layout
    def place():
      ti.root.place(y)

    y[None] = 1

