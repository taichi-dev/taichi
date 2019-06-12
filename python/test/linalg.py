import taichi_lang as ti
from pytest import approx

ti.cfg.arch = ti.cuda

def test_transpose():
  ti.reset()
  dim = 3
  m = ti.Matrix(dim, dim, ti.f32)

  @ti.layout
  def place():
    ti.root.place(m)

  @ti.kernel
  def transpose():
    mat = ti.transposed(m[None])
    m[None] = mat

  for i in range(dim):
    for j in range(dim):
      m(i, j)[None] = i * 2 + j * 7

  transpose()

  for i in range(dim):
    for j in range(dim):
      assert m(j, i)[None] == approx(i * 2 + j * 7)

