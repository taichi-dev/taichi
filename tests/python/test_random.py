import taichi as ti
from pytest import approx


@ti.all_archs
def test_random():
  n = 1024
  x = ti.var(ti.f32, shape=(n, n))

  @ti.kernel
  def fill():
    for i in range(n):
      for j in range(n):
        x[i, j] = ti.random()

  fill()
  X = x.to_numpy()
  for i in range(4):
    assert (X**i).mean() == approx(1 / (i + 1), rel=1e-2)
