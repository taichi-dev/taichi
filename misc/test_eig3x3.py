import taichi as ti 
ti.init()

@ti.kernel
def test():
    A = ti.Matrix([[3, 1, 1],[1, 2, 2], [1, 2, 2]])
    diagnal = ti.sym_eig(A)
    print(diagnal)

test()
