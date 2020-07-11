import taichi as ti

ti.init(print_preprocessed=True)


def func(x):
    print('hello', x)


@ti.kernel
def kern():
    func(233)


kern()
