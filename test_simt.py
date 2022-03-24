import taichi as ti

ti.init(arch=ti.cuda)


a = ti.field(dtype=ti.i32, shape=128)

@ti.kernel
def foo():
    ti.loop_config(block_dim=32)
    for i in range(128):
        a[i] = ti.lang.shfl_down_sync_i32(0x7FFFFFFF, i * 100, 2)
        # a[i] = i * 100


foo()
for i in range(32):
    print(a[i])
