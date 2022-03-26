import taichi as ti

ti.init()


@ti.kernel
def test_solve_3x3():
    A = ti.Matrix([[3.0, 2.0, -4.0], [2.0, 3.0, 3.0], [5.0, -3, 1.0]])
    b = ti.Vector([3.0, 15.0, 14.0])
    x = A.solve(b)
    print(x)


test_solve_3x3()
