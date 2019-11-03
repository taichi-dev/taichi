import taichi as ti

@ti.all_archs
def test_kernel_template_mapper():
  x = ti.var(ti.i32)
  y = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.place(x, y)

  mapper = ti.KernelTemplateMapper(3, (0, 1, 2))
  assert mapper.lookup((0, 0, 0)) == 0
  assert mapper.lookup((0, 1, 0)) == 1
  assert mapper.lookup((0, 0, 0)) == 0
  assert mapper.lookup((0, 0, 1)) == 2
  assert mapper.lookup((0, 1, 0)) == 1

  mapper = ti.KernelTemplateMapper(3, ())
  assert mapper.lookup((0, 0, 0)) == 0
  assert mapper.lookup((0, 1, 0)) == 0
  assert mapper.lookup((0, 0, 0)) == 0
  assert mapper.lookup((0, 0, 1)) == 0
  assert mapper.lookup((0, 1, 0)) == 0
  
  mapper = ti.KernelTemplateMapper(3, (1,))
  assert mapper.lookup((0, x, 0)) == 0
  assert mapper.lookup((0, y, 0)) == 1
  assert mapper.lookup((0, x, 1)) == 0

