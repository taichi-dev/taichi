import taichi as ti


@ti.test(require=ti.extension.quant)
def test_custom_float_unsigned():
    cu13 = ti.type_factory_.get_custom_int_type(13, False)
    exp = ti.type_factory_.get_custom_int_type(6, False)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [
        1 / 1024, 1.75 / 1024, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 128, 256,
        512, 1024
    ]

    for v in tests:
        x[None] = v
        assert x[None] == v


def main():
    ti.init()
    cu13 = ti.type_factory_.get_custom_int_type(13, True)
    exp = ti.type_factory_.get_custom_int_type(6, False)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    tests = [-2, -4, 6, 7, -6, -7, -8]

    for v in tests:
        x[None] = v
        print(v, x[None])
        assert x[None] == v


main()
