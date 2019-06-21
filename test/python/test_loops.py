import taichi_lang as ti
from pytest import approx

def test_loops():
  for arch in [ti.x86_64, ti.cuda]:
    ti.reset()
    ti.cfg.arch = arch
    x = ti.var(ti.f32)
    y = ti.var(ti.f32)

    N = 16
    @ti.layout
    def place():
      ti.root.dense(ti.i, N).place(x)
      ti.root.dense(ti.i, N).place(y)
      ti.root.lazy_grad()

    for i in range(1, N):
      y[i] = i - 5

    @ti.kernel
    def func():
      for i in range(1, N):
        x[i] = ti.abs(y[i])

    func()

    assert x[0] == abs(y[0])
    for i in range(1, N):
      assert x[i] == abs(y[i])
