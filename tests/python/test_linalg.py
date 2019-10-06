import taichi as ti
from pytest import approx

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


def test_polar_decomp():
  ti.reset()
  dim = 2
  m = ti.Matrix(dim, dim, ti.f32)
  r = ti.Matrix(dim, dim, ti.f32)
  s = ti.Matrix(dim, dim, ti.f32)
  I = ti.Matrix(dim, dim, ti.f32)
  D = ti.Matrix(dim, dim, ti.f32)

  @ti.layout
  def place():
    ti.root.place(m, r, s, I, D)

  @ti.kernel
  def polar():
    R, S = ti.polar_decompose(m[None])
    r[None] = R
    s[None] = S
    m[None] = R @ S
    I[None] = R @ ti.transposed(R)
    D[None] = S - ti.transposed(S)

  for i in range(dim):
    for j in range(dim):
      m(i, j)[None] = i * 2 + j * 7

  polar()

  for i in range(dim):
    for j in range(dim):
      assert m(i, j)[None] == approx(i * 2 + j * 7, abs=1e-5)
      assert I(i, j)[None] == approx(int(i == j), abs=1e-5)
      assert D(i, j)[None] == approx(0, abs=1e-5)

