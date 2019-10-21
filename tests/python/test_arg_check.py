import taichi as ti

def test_argument_error():
  ti.reset()
  x = ti.var(ti.i32)
  
  @ti.layout
  def layout():
    ti.root.place(x)
  
  try:
    @ti.kernel
    def set_i32_notype(v):
      pass
  except ti.KernelDefError:
    pass
    
  try:
    @ti.kernel
    def set_i32_args(*args):
      pass
  except ti.KernelDefError:
    pass
  
  try:
    @ti.kernel
    def set_i32_kwargs(**kwargs):
      pass
  except ti.KernelDefError:
    pass
  
  @ti.kernel
  def set_i32(v: ti.i32):
    x[None] = v
    
  set_i32(123)
  assert x[None] == 123


def test_ext_arr():
  ti.reset()
  N = 128
  x = ti.var(ti.f32)
  
  @ti.layout
  def layout():
    ti.root.dense(ti.i, N).place(x)
  
  @ti.kernel
  def set_f32(v: ti.ext_arr()):
    for i in range(N):
      x[i] = v[i] + i
  
  import numpy as np
  v = np.ones((N,), dtype=np.float32) * 10
  set_f32(v)
  for i in range(N):
    assert x[i] == 10 + i


