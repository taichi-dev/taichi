import taichi as ti

ti.init(ti.opengl, max_block_dim=8)
ti.set_logging_level(ti.DEBUG)

@ti.kernel
def func():
    for i in range(10):
        print(i)

func()
ti.sync()
