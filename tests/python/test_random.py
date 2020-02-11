import taichi as ti
from pytest import approx


def archs_support_random(func):
  return ti.archs_excluding(ti.metal)(func)


@archs_support_random
def test_random_float():
  for precision in [ti.f32, ti.f64]:
    ti.init()
    n = 1024
    x = ti.var(ti.f32, shape=(n, n))

    @ti.kernel
    def fill():
      for i in range(n):
        for j in range(n):
          x[i, j] = ti.random(precision)

    fill()
    X = x.to_numpy()
    for i in range(4):
      assert (X**i).mean() == approx(1 / (i + 1), rel=1e-2)

@archs_support_random
def test_random_int():
  for precision in [ti.i32, ti.i64]:
    ti.init()
    n = 1024
    x = ti.var(ti.f32, shape=(n, n))

    @ti.kernel
    def fill():
      for i in range(n):
        for j in range(n):
          v = ti.random(precision)
          if precision == ti.i32:
            x[i, j] = (float(v) + float(2 ** 31)) / float(2 ** 32)
          else:
            x[i, j] = (float(v) + float(2 ** 63)) / float(2 ** 64)

    fill()
    X = x.to_numpy()
    for i in range(4):
      assert (X**i).mean() == approx(1 / (i + 1), rel=1e-2)
