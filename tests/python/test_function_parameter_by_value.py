import taichi as ti

@ti.all_archs
def test_access_by_ref_should_crash():

  @ti.func
  def set_val(x, i):
    x = i

  ret = ti.var(ti.i32, shape=())

  @ti.kernel
  def task():
    set_val(ret[None], 112)

  task()
  print(ret[None])
