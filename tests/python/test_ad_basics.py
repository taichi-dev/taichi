import taichi as ti

def grad_test1(llvm):
  ti.reset()
  ti.cfg.use_llvm = llvm

  x = ti.var(ti.i32)

  @ti.layout
  def place():
    ti.root.dense(ti.i, 1).place(x)

  @ti.kernel
  def func():
    x[0] = 1

  func()
  
grad_test1(False)
grad_test1(True)
grad_test1(True)

