import taichi as ti


# @ti.test(require=ti.extension.quant)
# def test_shared_exponents():
def main():
    ti.init()
    cit = ti.type_factory.custom_int(8, True)
    exp = ti.type_factory.custom_int(8, False)
    cft = ti.type_factory.custom_float(significand_type=cit,
                                       exponent_type=exp,
                                       scale=1)
    a = ti.field(dtype=cft)
    b = ti.field(dtype=cft)
    ti.root._bit_struct(num_bits=32).place(a, b, shared_exponent=True)

    ti.get_runtime().materialize()
    ti.get_runtime().prog.print_snode_tree()


main()
