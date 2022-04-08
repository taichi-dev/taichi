import taichi as ti

ti.init(ti.vulkan)

x = ti.ndarray(ti.i32, shape=(4, 4))


@ti.kernel
def fill(x: ti.types.any_arr(), base: int):
    for i, j in x:
        x[i, j] = i * base + j


def run():
    fill(x, 100)
    print(x.to_numpy())


def aot():
    m = ti.aot.Module(ti.vulkan)
    m.add_kernel(fill, (x, ))
    m.save('./generated', 'demo')


if __name__ == '__main__':
    # run()
    aot()
