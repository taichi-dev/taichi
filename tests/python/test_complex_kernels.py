import taichi as ti

def complex_kernel(func):
  def decorated(*args, **kwargs):
    ti.get_runtime().inside_complex_kernel = True
    ti.get_runtime().target_tape.insert(decorated, args)
    try:
      func(*args, **kwargs)
    except Exception as e:
      raise e
    finally:
      ti.get_runtime().inside_complex_kernel = False
  decorated.grad = None
  return decorated

def complex_kernel_grad(primal):
  def decorator(func):
    def decorated(*args, **kwargs):
      func(*args, **kwargs)
    primal.grad = decorated
    return decorated
  return decorator

ti.complex_kernel = complex_kernel
ti.complex_kernel_grad = complex_kernel_grad

@ti.all_archs
def test_complex_kernels():
  x = ti.var(ti.f32)
  total = ti.var(ti.f32)

  n = 128

  @ti.layout
  def place():
    ti.root.dense(ti.i, n).place(x)
    ti.root.place(total)
    ti.root.lazy_grad()

  @ti.kernel
  def func(mul: ti.f32):
    for i in range(n):
      ti.atomic_add(total[None], x[i] * mul)

  @ti.complex_kernel
  def forward(mul):
    func(mul, extra_frame_backtrace=2)

  @ti.complex_kernel_grad(forward)
  def backward(mul, **kwargs):
    func.grad(mul, extra_frame_backtrace=4)

  with ti.Tape(loss=total):
    forward(4)
  assert x[0] == 45


test_complex_kernels()