import taichi as ti


def test_simple_array():
    ti.init(arch=ti.cpu, print_ir=True, cfg_optimization=False)
    ci1 = ti.type_factory_.get_custom_int_type(1, False)

    x = ti.field(dtype=ci1)

    ti.root._bit_array(ti.i, 32, num_bits=32).place(x)

    ti.get_runtime().materialize()

    @ti.kernel
    def set_val():
        for i in range(32):
            x[i] = i % 2

    @ti.kernel
    def verify_val():
        for i in range(32):
            assert x[i] == i % 2

    set_val()
    verify_val()


def test_2D_array():
    ti.init(arch=ti.cpu, print_ir=True)
    ci1 = ti.type_factory_.get_custom_int_type(1, True)

    x = ti.field(dtype=ci1)

    ti.root._bit_array(ti.ij, (4, 8), num_bits=32).place(x)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()


test_simple_array()
