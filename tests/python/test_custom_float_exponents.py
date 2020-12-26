import taichi as ti
import math
from pytest import approx

# @ti.test(require=ti.extension.quant)
# def test_custom_float():


def main():
    ti.init(print_ir=True)
    # Note: digits_type must be unsigned with using exponent
    cu13 = ti.type_factory_.get_custom_int_type(13, False)
    exp = ti.type_factory_.get_custom_int_type(6, True)
    cft = ti.type_factory.custom_float(significand_type=cu13,
                                       exponent_type=exp,
                                       scale=1)
    x = ti.field(dtype=cft)

    ti.root._bit_struct(num_bits=32).place(x)

    ti.get_runtime().materialize()
    ti.get_runtime().print_snode_tree()

    for v in [3, 4, 5, 6, 7, 128, 256, 512, 1024]:
        x[None] = v
        print(x[None])
        assert x[None] == v


main()
