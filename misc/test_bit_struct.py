import taichi as ti


def test_simple_array():
    ti.init(arch=ti.cpu, print_ir=True)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu19 = ti.type_factory_.get_custom_int_type(19, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu19)

    N = 1024

    ti.root.dense(ti.i, N)._bit_struct(num_bits=32).place(x, y)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()

    @ti.kernel
    def foo():
        for i in range(N):
            print(x[i])
            print(y[i])

    foo()


def test_simple_singleton():
    ti.init(arch=ti.cpu, debug=True, print_ir=True, advanced_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu14 = ti.type_factory_.get_custom_int_type(14, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu14)

    ti.root._bit_struct(num_bits=32).place(x, y)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()

    @ti.kernel
    def foo():
        x[None] = 2**13-1
        assert(x[None] == 2**13-1)
        y[None] = 2**14-1
        assert(int(y[None]) == 2**14-1)

    foo()


test_simple_singleton()
