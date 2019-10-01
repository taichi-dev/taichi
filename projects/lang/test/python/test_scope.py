import taichi as ti

ti.runtime.print_preprocessed = True

def test_scope():
  # In the future the following code should throw an exception at the python front end
  # instead of crashing the compiler
  return
  for arch in [ti.x86_64, ti.cuda]:
    # ti.reset()
    ti.cfg.arch = arch
    x = ti.var(ti.f32)

    N = 1
    @ti.layout
    def place():
      ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
      if 1 > 0:
        val = 1

      ti.print(val)

    func()

