import taichi as ti

@ti.all_archs
def test_1d():
  x = ti.var(ti.f32, shape=(16))

  @ti.kernel
  def func():
    for i in ti.ndrange((4, 10)):
      x[i] = i

  func()
  
  for i in range(16):
    if 4 <= i < 10:
      assert x[i] == i
    else:
      assert x[i] == 0


@ti.all_archs
def test_2d():
  # ti.get_runtime().print_preprocessed = True
  # ti.cfg.print_ir = True
  x = ti.var(ti.f32, shape=(16, 32))
  
  t = 8
  
  @ti.kernel
  def func():
    for i, j in ti.ndrange((4, 10), (3, t)):
      val = i + j * 10
      x[i, j] = val
  
  func()
  for i in range(16):
    for j in range(32):
      if 4 <= i < 10 and 3 <= j < 8:
        assert x[i, j] == i + j * 10
      else:
        assert x[i, j] == 0
        
@ti.all_archs
def test_3d():
  # ti.get_runtime().print_preprocessed = True
  # ti.cfg.print_ir = True
  x = ti.var(ti.f32, shape=(16, 32, 64))
  
  @ti.kernel
  def func():
    for i, j, k in ti.ndrange((4, 10), (3, 8), 17):
      x[i, j, k] = i + j * 10 + k * 100
  
  func()
  for i in range(16):
    for j in range(32):
      for k in range(64):
        if 4 <= i < 10 and 3 <= j < 8 and k < 17:
          assert x[i, j, k] == i + j * 10 + k * 100
        else:
          assert x[i, j, k] == 0
