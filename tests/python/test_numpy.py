import taichi as ti
import numpy as np

@ti.all_archs
def test_numpy():
  val = ti.var(ti.i32)

  n = 4

  @ti.layout
  def values():
    ti.root.dense(ti.i, n).place(val)

  @ti.kernel
  def test_numpy(arr: np.ndarray):
    for i in range(n):
      arr[i] = arr[i] ** 2

  a = np.array([4, 8, 1, 24], dtype=np.float32)
  
  for i in range(n):
    a[i] = i * 2

  test_numpy(a)
  
  for i in range(n):
    assert a[i] == i * i * 4
