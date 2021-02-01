class TypeFactory:
    def __init__(self):
        from taichi.core import ti_core
        self.core = ti_core.get_type_factory_instance()

    def custom_int(self, bits, signed=True, compute_type=None):
        import taichi as ti
        if compute_type is None:
            compute_type = ti.get_runtime().default_ip
        if isinstance(compute_type, ti.core.DataType):
            compute_type = compute_type.get_ptr()
        return self.core.get_custom_int_type(bits, signed, compute_type)

    def custom_float(self,
                     significand_type,
                     exponent_type=None,
                     compute_type=None,
                     scale=1.0):
        import taichi as ti
        if compute_type is None:
            compute_type = ti.get_runtime().default_fp
        if isinstance(compute_type, ti.core.DataType):
            compute_type = compute_type.get_ptr()
        return self.core.get_custom_float_type(significand_type,
                                               exponent_type,
                                               compute_type,
                                               scale=scale)


class Quant:
    @staticmethod
    def int(bits, signed=False, compute=None):
        import taichi as ti
        if compute is None:
            compute = ti.get_runtime().default_ip
        return ti.type_factory.custom_int(bits, signed, compute)

    @staticmethod
    def fixed(frac, signed=True, range=1.0, compute=None):
        import taichi as ti
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=ti.i32)
        if signed:
            scale = range / 2**(frac - 1)
        else:
            scale = range / 2**frac
        if compute is None:
            compute = ti.get_runtime().default_fp
        return ti.type_factory.custom_float(frac_type, None, compute, scale)

    @staticmethod
    def float(exp, frac, signed=True, compute=None):
        import taichi as ti
        # Exponent is always unsigned
        exp_type = Quant.int(bits=exp, signed=False, compute=ti.i32)
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=ti.i32)
        if compute is None:
            compute = ti.get_runtime().default_fp
        return ti.type_factory.custom_float(significand_type=frac_type,
                                            exponent_type=exp_type,
                                            compute_type=compute)
