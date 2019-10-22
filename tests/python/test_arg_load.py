import taichi as ti

def test_arg_load():
  ti.reset()
  x = ti.var(ti.i32)
  y = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.place(x, y)

  @ti.kernel
  def set_i32(v : ti.i32):
    x[None] = v

  @ti.kernel
  def set_f32(v : ti.f32):
    y[None] = v

  set_i32(123)
  assert x[None] == 123

  set_i32(456)
  assert x[None] == 456

  set_f32(0.125)
  assert y[None] == 0.125

  set_f32(1.5)
  assert y[None] == 1.5


def test_ext_arr():
  ti.reset()
  N = 128
  x = ti.var(ti.f32)

  @ti.layout
  def layout():
    ti.root.dense(ti.i, N).place(x)

  @ti.kernel
  def set_f32(v: ti.ext_arr()):
    for i in range(N):
      x[i] = v[i] + i

  import numpy as np
  v = np.ones((N,), dtype=np.float32) * 10
  set_f32(v)
  for i in range(N):
    assert x[i] == 10 + i

