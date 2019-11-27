import taichi as ti
import numpy as np

def with_data_type(dt):
  val = ti.var(ti.i32)

  n = 4

  @ti.layout
  def values():
    ti.root.dense(ti.i, n).place(val)

  @ti.kernel
  def test_numpy(arr: ti.ext_arr()):
    for i in range(n):
      arr[i] = arr[i] ** 2

  a = np.array([4, 8, 1, 24], dtype=dt)

  for i in range(n):
    a[i] = i * 2

  test_numpy(a)

  for i in range(n):
    assert a[i] == i * i * 4

@ti.all_archs
def test_numpy_f32():
  with_data_type(np.float32)

@ti.all_archs
def test_numpy_f64():
  with_data_type(np.float64)

@ti.all_archs
def test_numpy_i32():
  with_data_type(np.int32)

@ti.all_archs
def test_numpy_i64():
  with_data_type(np.int64)
