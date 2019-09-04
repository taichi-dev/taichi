import taichi_lang as ti
import taichi as tc

tc.set_gdb_trigger(True)

# x, y = ti.var(ti.f32), ti.var(ti.f32)
# z, w = ti.var(ti.f32), ti.var(ti.f32)
val = ti.var(ti.i32)
y = ti.var(ti.i32)

ti.cfg.use_llvm = True
ti.cfg.print_ir = True
# ti.cfg.print_struct_llvm_ir = True
ti.cfg.print_kernel_llvm_ir = True

# ti.cfg.print_ir = True
# ti.runtime.print_preprocessed = True
ti.cfg.arch = ti.cuda

n = 32

@ti.layout
def values():
  ti.root.dense(ti.i, 4).place(val)

val[0] = 123456
print('val', val[0])

for i in range(4):
  val[i] = (i + 1) * 2 + 12
  print(val[i])


@ti.kernel
def test():
  for i in range(4):
    val[i] = i * 10

test()
