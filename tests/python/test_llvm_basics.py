import taichi as ti

def test_simle():
  return
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
      x[7] = 120
    
    func()
    
    for i in range(n):
      if i == 7:
        assert x[i] == 0
      else:
        assert x[i] == 120

def test_range_loops():
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


