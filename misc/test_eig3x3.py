import taichi as ti

ti.init()


@ti.kernel
def test():
    A = ti.Matrix([[3.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0]],
                  dt=ti.f32)
    diagnal, Q = ti.sym_eig(A)
    print(diagnal, Q)


test()
