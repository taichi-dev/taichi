import taichi as ti
from pytest import approx

def test_clear():
  return
  ti.reset()
  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)

  x[0] = 1
  assert x[0] == 1
  x.clear()
  assert x[0] == 0

def test_nested_subscript():
  ti.reset()

  x = ti.var(ti.i32)
  y = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)
    ti.root.dense(ti.i, 1).place(y)

  x[0] = 0

  @ti.kernel
  def inc():
    for i in range(1):
      x[x[i]] += 1

  inc()

  assert x[0] == 1


@ti.all_archs
def test_norm():
  val = ti.var(ti.i32)
  f = ti.var(ti.f32)
  
  n = 1024
  
  @ti.layout
  def values():
    ti.root.dense(ti.i, n).dense(ti.i, 2).place(val, f)
  
  @ti.kernel
  def test():
    for i in range(n):
      s = 0
      for j in range(10):
        s += j
      a = ti.Vector([0.4, 0.3])
      val[i] = s + ti.cast(a.norm() * 100, ti.i32) + i
  
  test()
  
  @ti.kernel
  def test2():
    for i in range(n):
      val[i] += 1
  
  test2()
  
  for i in range(n):
    assert val[i] == 96 + i
    

@ti.all_archs
def test_simple2():
  val = ti.var(ti.i32)
  f = ti.var(ti.f32)
  
  n = 16
  
  @ti.layout
  def values():
    ti.root.dense(ti.i, n).place(val, f)
  
  @ti.kernel
  def test():
    for i in range(n):
      val[i] = i * 2
  
  test()
  
  @ti.kernel
  def test2():
    for i in range(n):
      val[i] += 1
  
  test2()
  
  for i in range(n):
    assert val[i] == 1 + i * 2

