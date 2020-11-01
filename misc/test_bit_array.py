import taichi as ti


def test_simple_array():
    ti.init(arch=ti.cpu, print_ir=True)
    ci1 = ti.type_factory_.get_custom_int_type(1, True)

    x1 = ti.field(dtype=ci1)
    x2 = ti.field(dtype=ci1)
    x3 = ti.field(dtype=ci1)
    x4 = ti.field(dtype=ci1)
    x5 = ti.field(dtype=ci1)

    N = 1024

    ti.root.dense(ti.i, N)._bit_array(num_bits=32).place(x1, x2, x3, x4, x5)

    ti.get_runtime().print_snode_tree()
    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()


test_simple_array()
