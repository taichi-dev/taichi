import taichi as ti

def grad_test1():
  ti.reset()
  ti.cfg.use_llvm = True
  # ti.cfg.arch = ti.cuda

  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    x[0] = 0

  # func.materialize()
  func()


def grad_test2():
  print('grad_test2')
  ti.set_gdb_trigger()
  ti.reset()
  ti.lang_core.test_throw()
  ti.cfg.use_llvm = True

  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    x[0] = 1

  # func.materialize()
  func()


grad_test1()
grad_test2()
