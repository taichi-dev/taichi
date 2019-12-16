import taichi as ti


@ti.all_archs
def test_fill_scalar():
  val = ti.var(ti.i32)

  n = 4
  m = 7

  @ti.layout
  def values():
    ti.root.dense(ti.ij, (n, m)).place(val)

  for i in range(n):
    for j in range(m):
      val[i, j] = i + j * 3

  val.fill(2)

  for i in range(n):
    for j in range(m):
      assert val[i, j] == 2


@ti.all_archs
def test_fill_matrix_scalar():
  val = ti.Vector(2, 3, ti.i32)

  n = 4
  m = 7

  @ti.layout
  def values():
    ti.root.dense(ti.ij, (n, m)).place(val)

  for i in range(n):
    for j in range(m):
      for p in range(2):
        for q in range(3):
          val[i, j][p, q] = i + j * 3

  val.fill(2)

  for i in range(n):
    for j in range(m):
      for p in range(2):
        for q in range(3):
          assert val[i, j][p, q] == 2


@ti.all_archs
def test_fill_matrix_matrix():
  val = ti.Vector(2, 3, ti.i32)

  n = 4
  m = 7

  @ti.layout
  def values():
    ti.root.dense(ti.ij, (n, m)).place(val)

  for i in range(n):
    for j in range(m):
      for p in range(2):
        for q in range(3):
          val[i, j][p, q] = i + j * 3

  mat = ti.Matrix([[0, 1, 2], [2, 3, 4]])

  val.fill(mat)

  for i in range(n):
    for j in range(m):
      for p in range(2):
        for q in range(3):
          assert val[i, j][p, q] == mat.get_entry(p, q)
