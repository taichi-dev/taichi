import taichi as ti

def test_linear():
  ti.reset()
  
  ti.core.test_throw()
  
  ti.cfg.use_llvm = True

  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)
    ti.root.dense(ti.i, n).place(y)

  @ti.kernel
  def func():
    for i in range(n):
      x[i] = i
      y[i] = i + 123


  ti.core.test_throw()
  # ti.runtime.materialize()
  ti.core.test_throw()
  func()
  ti.core.test_throw()
  

  for i in range(n):
    assert x[i] == i
    assert y[i] == i + 123
  
def test_linear_repeated():
  for i in range(10):
    test_linear()
    
test_linear_repeated()
  
def test_linear_nested():
  ti.reset()
  ti.cfg.use_llvm = True

  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x)
    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(y)

  for i in range(n):
    x[i] = i
    y[i] = i + 123

  for i in range(n):
    assert x[i] == i
    assert y[i] == i + 123

def test_2d_nested():
  ti.reset()
  ti.cfg.use_llvm = True

  x = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.ij, n // 16).dense(ti.ij, (32, 16)).place(x)

  for i in range(n * 2):
    for j in range(n):
      x[i, j] = i + j * 10

  for i in range(n * 2):
    for j in range(n):
      assert x[i, j] == i + j * 10
