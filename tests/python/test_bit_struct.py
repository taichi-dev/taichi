import taichi as ti


# @ti.test()
def test_simple():
    ti.init(arch=ti.cpu)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu19 = ti.type_factory_.get_custom_int_type(19, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu19)

    N = 1024

    ti.root.dense(ti.i, N)._bit_struct(num_bits=32).place(x, y)

    ti.get_runtime().print_snode_tree()
    
    @ti.kernel
    def foo():
        for i in range(N):
            print(x[i])
            print(y[i])

    foo()


# test_simple()
