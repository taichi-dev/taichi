from taichi.lang import impl
from taichi.lang import type_factory_impl as tf_impl

import taichi as ti


class Quant:
    @staticmethod
    def int(bits, signed=False, compute=None):
        if compute is None:
            compute = impl.get_runtime().default_ip
        return tf_impl.type_factory.custom_int(bits, signed, compute)

    @staticmethod
    def fixed(frac, signed=True, range=1.0, compute=None):
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=ti.i32)
        if signed:
            scale = range / 2**(frac - 1)
        else:
            scale = range / 2**frac
        if compute is None:
            compute = impl.get_runtime().default_fp
        return tf_impl.type_factory.custom_float(frac_type, None, compute,
                                                 scale)

    @staticmethod
    def float(exp, frac, signed=True, compute=None):
        # Exponent is always unsigned
        exp_type = Quant.int(bits=exp, signed=False, compute=ti.i32)
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=ti.i32)
        if compute is None:
            compute = impl.get_runtime().default_fp
        return tf_impl.type_factory.custom_float(significand_type=frac_type,
                                                 exponent_type=exp_type,
                                                 compute_type=compute)


# Unstable API
quant = Quant
