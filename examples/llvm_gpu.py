import taichi as ti
import taichi as tc

tc.set_gdb_trigger(True)

val = ti.var(ti.i32)
f = ti.var(ti.f32)

ti.cfg.use_llvm = True
ti.cfg.arch = ti.cuda
# ti.cfg.print_ir = True
# ti.cfg.print_kernel_llvm_ir = True

n = 32

@ti.layout
def values():
  ti.root.dense(ti.i, 4).place(val, f)

@ti.kernel
def test():
  for i in range(4):
    # ti.print(i)
    val[i] = i * 10
    ti.print(val[i])
    f[i] = ti.sin(ti.cast(i, ti.f32))
    ti.print(f[i])

for i in range(10):
  test()
for i in range(4):
  print(val[i])
  print(f[i])
