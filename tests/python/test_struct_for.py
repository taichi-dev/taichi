import taichi as ti


@ti.program_test
def test_linear():
  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)
    ti.root.dense(ti.i, n).place(y)

  @ti.kernel
  def fill():
    for i in x:
      x[i] = i
      y[i] = i * 2
      
  fill()

  for i in range(n):
    assert x[i] == i
    assert y[i] == i * 2
    
@ti.program_test
def test_nested():
  x = ti.var(ti.i32)
  y = ti.var(ti.i32)
  
  n = 128
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n // 4).dense(ti.i, 4).place(x)
    ti.root.dense(ti.i, n).place(y)
  
  @ti.kernel
  def fill():
    for i in x:
      x[i] = i
      y[i] = i * 2
  
  fill()
  
  for i in range(n):
    assert x[i] == i
    assert y[i] == i * 2
    
@ti.program_test
def test_nested2():
  x = ti.var(ti.i32)
  y = ti.var(ti.i32)
  
  n = 2048
  
  @ti.layout
  def place():
    ti.root.dense(ti.i, n // 512).dense(ti.i, 16).dense(ti.i, 8).dense(ti.i, 4).place(x)
    ti.root.dense(ti.i, n).place(y)
  
  @ti.kernel
  def fill():
    for i in x:
      x[i] = i
      y[i] = i * 2
  
  fill()
  
  for i in range(n):
    assert x[i] == i
    assert y[i] == i * 2
    
@ti.program_test
def test_linear_k():
  x = ti.var(ti.i32)
  
  n = 128
  
  @ti.layout
  def place():
    ti.root.dense(ti.k, n).place(x)
  
  @ti.kernel
  def fill():
    for i in x:
      x[i] = i
      
  fill()
  
  for i in range(n):
    assert x[i] == i
