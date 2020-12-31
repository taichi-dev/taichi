import taichi as ti


# @ti.test(require=ti.extension.quant)
# def test_shared_exponents():
def main():
    ti.init()
    exp = ti.type_factory.custom_int(8, False)
    cit = ti.type_factory.custom_int(12, True)
    cft = ti.type_factory.custom_float(significand_type=cit,
                                       exponent_type=exp,
                                       scale=1)
    a = ti.field(dtype=cft)
    b = ti.field(dtype=cft)
    ti.root._bit_struct(num_bits=32).place(a, b, shared_exponent=True)

    ti.get_runtime().materialize()
    ti.get_runtime().prog.print_snode_tree()

    a[None] = 3.2
    print(a[None], b[None])
    b[None] = 0.25
    print(a[None], b[None])


main()
