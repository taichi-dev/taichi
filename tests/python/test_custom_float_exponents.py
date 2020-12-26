import taichi as ti
import math
from pytest import approx

# @ti.test(require=ti.extension.quant)
# def test_custom_float():


def main():
    ti.init(print_ir=True)
    # Note: digits_type must be unsigned with using exponent
    cu13 = ti.type_factory_.get_custom_int_type(13, False)
    exp = ti.type_factory_.get_custom_int_type(6, False)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()

    for v in [1 / 1024, 1.75 / 1024, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 128, 256, 512, 1024]:
        x[None] = v
        print(v, x[None])
        assert x[None] == v


main()
