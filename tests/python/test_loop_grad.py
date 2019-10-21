import taichi as ti

def test_loop_grad():
  for arch in [ti.x86_64, ti.cuda]:
    ti.reset()
    ti.cfg.arch = arch
    x = ti.var(ti.f32)

    n = 16
    m = 8
    @ti.layout
    def place():
      ti.root.dense(ti.ij, (n, m)).place(x)
      ti.root.lazy_grad()

    @ti.kernel
    def func():
      for k in range(n):
        for i in range(m - 1):
          x[k, i + 1] = x[k, i] * 2


    for k in range(n):
      x[k, 0] = k
    func()

    for k in range(n):
      x.grad[k, m - 1] = 1
    func.grad()

    for k in range(n):
      for i in range(m):
        assert x[k, i] == 2 ** i * k
        assert x.grad[k, i] == 2 ** (m - 1 - i)

