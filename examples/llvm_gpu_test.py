import taichi as ti

def test_llvm_gpu():
  ti.reset()
  
  val = ti.var(ti.i32)
  f = ti.var(ti.f32)

  ti.cfg.use_llvm = True
  ti.cfg.arch = ti.cuda
  # ti.cfg.print_ir = True
  # ti.cfg.print_kernel_llvm_ir = True

  n = 16

  @ti.layout
  def values():
    ti.root.dense(ti.i, n).place(val, f)

  @ti.kernel
  def test():
    for i in range(n):
      # ti.print(i)
      val[i] = i * 2

  test()
  
  @ti.kernel
  def test2():
    for i in range(n):
      val[i] += 1
  
  test2()

  for i in range(n):
    # print(i, val[i], f[i])
    assert val[i] == 1 + i * 2

test_llvm_gpu()
