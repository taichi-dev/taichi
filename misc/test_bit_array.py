import taichi as ti


def test_simple_array():
    ti.init(arch=ti.cpu, print_ir=True)
    ci1 = ti.type_factory_.get_custom_int_type(1, True)

    x = ti.field(dtype=ci1)

    ti.root._bit_array(ti.i, 32, num_bits=32).place(x)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()


def test_2D_array():
    ti.init(arch=ti.cpu, print_ir=True)
    ci1 = ti.type_factory_.get_custom_int_type(1, True)

    x = ti.field(dtype=ci1)

    ti.root._bit_array(ti.ij, (4, 8), num_bits=32).place(x)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()


test_2D_array()
