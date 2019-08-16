import taichi_lang as ti
import taichi as tc
import numpy as np

tc.set_gdb_trigger(True)

val = ti.var(ti.i32)

ti.cfg.print_ir = True

# ti.cfg.print_struct_llvm_ir = True

n = 32

@ti.layout
def values():
  ti.root.dense(ti.i, 4).place(val)

@ti.kernel
def test_numpy(arr: np.ndarray):
  for i in range(4):
    val[i] = i * 20
    ti.print(arr[i])

a = np.array([4, 8, 1, 24], dtype=np.float32)
test_numpy(a)
