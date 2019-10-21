import taichi as ti

x = ti.var(ti.i32)
y = ti.var(ti.f32)

@ti.layout
def layout():
  ti.root.place(x, y)

def test_arg_load():
  ti.reset()
  x = ti.var(ti.i32)
  y = ti.var(ti.f32)
  
  @ti.layout
  def layout():
    ti.root.place(x, y)
  
  @ti.kernel
  def set_i32(v: ti.i32):
    x[None] = v
  
  @ti.kernel
  def set_f32(v: ti.f32):
    y[None] = v
  
  success = False
  try:
    set_i32(123)
  except Exception as e:
    assert type(e) == ti.KernelArgError
    success = True
  assert success
  
  assert x[None] == 123
  
  set_i32(456)
  assert x[None] == 456
  
  set_f32(0.125)
  assert y[None] == 0.125
  
  set_f32(1.5)
  assert y[None] == 1.5


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
  
test_argument_error()


def test_ext_arr():
  ti.reset()
  N = 128
  x = ti.var(ti.i32)
  
  @ti.layout
  def layout():
    ti.root.dense(ti.i, N).place(x)
  
  @ti.kernel
  def set_f32(v: ti.ext_arr()):
    for i in range(N):
      x[i] = v[i]
  
  set_f32()
  
test_ext_arr()

