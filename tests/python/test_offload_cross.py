import taichi as ti


@ti.all_archs
def test_offload_with_cross_block_locals():
  ret = ti.var(ti.f32)

  @ti.layout
  def place():
    ti.root.place(ret)

  @ti.kernel
  def ker():
    s = 0
    for i in range(10):
      s += i
    ret[None] = s

  ker()

  assert ret[None] == 45


@ti.all_archs
def test_offload_with_cross_block_locals2():
  ret = ti.var(ti.f32)

  @ti.layout
  def place():
    ti.root.place(ret)

  @ti.kernel
  def ker():
    s = 0
    for i in range(10):
      s += i
    ret[None] = s
    s = ret[None] * 2
    for i in range(10):
      ti.atomic_add(ret[None], s)

  ker()

  assert ret[None] == 45 * 21
