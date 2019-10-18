import taichi as ti

def test_loops():
  for arch in [ti.x86_64, ti.cuda]:
    ti.reset()
    ti.cfg.use_llvm = True
    ti.cfg.arch = arch
    x = ti.var(ti.i32)

    n = 128

    @ti.layout
    def place():
      ti.root.dense(ti.i, n).place(x)

    @ti.kernel
    def func():
      for i in range(n):
        x[i] = i + 123
        
    func()

    for i in range(n):
      assert x[i] == i + 123
    

