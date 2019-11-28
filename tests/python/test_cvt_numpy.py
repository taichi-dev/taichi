import taichi as ti
import numpy as np

@ti.all_archs
def test_numpy_2d():
  val = ti.var(ti.i32)

  n = 4
  m = 7

  @ti.layout
  def values():
    ti.root.dense(ti.ij, (n, m)).place(val)

  for i in range(n):
    for j in range(m):
      val[i, j] = i + j * 3

  arr = val.to_numpy()

  assert arr.shape == (4, 7)
  for i in range(n):
    for j in range(m):
      assert arr[i, j] == i + j * 3
