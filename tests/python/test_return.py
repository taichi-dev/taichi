import taichi as ti


@ti.must_throw(ti.TaichiSyntaxError)
def test_return_in_kernel():

  @ti.kernel
  def kernel():
    return 1

  kernel()


@ti.all_archs
def test_return_type():

  @ti.func
  def foo(x):
    if x > 0:
      return int(x)
    else:
      return float(x)

  @ti.kernel
  def kernel():
    assert foo(2.6) == 2
    assert foo(-2.4) == -2.4

  kernel()
