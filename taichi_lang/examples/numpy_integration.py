import taichi_lang as ti
import taichi as tc
import numpy as np

tc.set_gdb_trigger(True)

val = ti.var(ti.i32)

n = 32

@ti.layout
def values():
  ti.root.dense(ti.i, 4).place(val)

@ti.kernel
def test_numpy(arr: np.ndarray):
  for i in range(4):
    ti.print(arr[i])
    arr[i] = i * i

a = np.array([4, 8, 1, 24], dtype=np.float32)
test_numpy(a)
print(a)
