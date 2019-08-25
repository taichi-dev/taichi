import taichi_lang as ti
import taichi as tc

tc.set_gdb_trigger(True)

# x, y = ti.var(ti.f32), ti.var(ti.f32)
# z, w = ti.var(ti.f32), ti.var(ti.f32)
val = ti.var(ti.i32)
y = ti.var(ti.i32)

ti.cfg.use_llvm = True
ti.cfg.print_struct_llvm_ir = True

# ti.cfg.print_ir = True
# ti.runtime.print_preprocessed = True

n = 32

@ti.layout
def values():
  # fork = ti.root.dense(ti.k, 128)
  # fork.dense(ti.ij, 16).place(x, y)
  # fork.dense(ti.ijk, 4).dense(ti.i, 8).place(z, w)
  ti.root.dense(ti.i, 4).place(val)
  ti.root.dense(ti.i, 4).dense(ti.i, 4).place(y)

val[0] = 123456
print('val', val[0])

for i in range(4):
  val[i] = (i + 1) * 2 + 12
  print(val[i])


@ti.kernel
def test():
  a = 0
  for i in range(4):
    val[i] = i * 10
    ti.print(val[i])
    if i % 2 == 0:
      a += i
    ti.print(a)

@ti.kernel
def test1():
  for i in range(4):
    val[i] = i * 20
    ti.print(val[i])

@ti.kernel
def test_struct():
  for i in y:
    val[i] = i
    ti.print(val[i])

# test()
# test1()
test_struct()
