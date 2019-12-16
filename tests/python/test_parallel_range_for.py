import taichi as ti


@ti.all_archs
def test_parallel_range_for():
  n = 1024 * 1024
  val = ti.var(ti.i32, shape=(n))

  @ti.kernel
  def fill():
    ti.parallelize(8)
    ti.block_dim(8)
    for i in range(n):
      val[i] = i

  fill()

  for i in range(n):
    assert val[i] == i
