import taichi_lang as ti

def test_while():
  for arch in [ti.x86_64, ti.cuda]:
    ti.reset()
    ti.cfg.arch = arch
    x = ti.var(ti.f32)

    N = 1
    @ti.layout
    def place():
      ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
      i = 0
      s = 0
      while i < 10:
        s += i
        i += 1
      x[0] = s

    func()
    assert x[0] == 45
