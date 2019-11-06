import taichi as ti

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
    func(mul)
    func(mul)

  @ti.complex_kernel_grad(forward)
  def backward(mul):
    func.grad(mul)

  with ti.Tape(loss=total):
    forward(4)
  for i in range(n):
    assert x.grad[0] == 4
