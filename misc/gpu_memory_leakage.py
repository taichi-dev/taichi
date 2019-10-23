# The leakage is due to random number generator state variables.
# Not fixing this since we are switching from CUDA to llvm.

import taichi as ti

def test_gpu_memory():
  ti.reset()
  
  val = ti.var(ti.i32)
  f = ti.var(ti.f32)

  # ti.cfg.use_llvm = True
  ti.cfg.arch = ti.cuda
  # ti.cfg.print_ir = True
  # ti.cfg.print_kernel_llvm_ir = True

  n = 1024

  @ti.layout
  def values():
    ti.root.dense(ti.i, n).place(val, f)

  @ti.kernel
  def test():
    f[0] = 0

  test()
  while True:pass
  
while True:
  test_gpu_memory()
  
