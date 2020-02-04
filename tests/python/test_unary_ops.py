import taichi as ti
import numpy as np

@ti.all_archs
def _test_op(dt, taichi_op, np_op):
  n = 4
  val = ti.var(dt, shape=n)
  ti.get_runtime().default_fp = dt

  def f(i):
    return i * 0.1 + 0.4

  @ti.kernel
  def fill():
    for i in range(n):
      val[i] = taichi_op(f(ti.cast(i, dt)))

  fill()

  # check that it is double precision
  for i in range(n):
    if dt == ti.f64:
      assert abs(np_op(float(f(i))) - val[i]) < 1e-15
    else:
      assert abs(np_op(float(f(i))) - val[i]) < 1e-6


def test_f64_trig():
  for dt in [ti.f32, ti.f64]:
    _test_op(dt, ti.sin, np.sin)
    _test_op(dt, ti.cos, np.cos)
    _test_op(dt, ti.asin, np.arcsin)
    _test_op(dt, ti.acos, np.arccos)
    _test_op(dt, ti.tan, np.tan)
    _test_op(dt, ti.tanh, np.tanh)
    _test_op(dt, ti.exp, np.exp)
    _test_op(dt, ti.log, np.log)
