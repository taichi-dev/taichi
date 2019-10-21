import taichi as ti

def test_loop_grad():
  for arch in [ti.x86_64, ti.cuda]:
    ti.reset()
    ti.cfg.arch = arch
    x = ti.var(ti.f32)

    N = 8
    @ti.layout
    def place():
      ti.root.dense(ti.i, N).place(x)
      ti.root.lazy_grad()


    @ti.kernel
    def func():
      for k in range(1):
        for i in range(N - 1):
          x[i + 1] = x[i] * 2

    x[0] = 3
    func()
    x.grad[N - 1] = 1
    func.grad()
    assert x[N - 1] == 384
    assert x.grad[0] == 128

