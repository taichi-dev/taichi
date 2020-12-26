class TypeFactory:
    def __init__(self):
        from taichi.core import ti_core
        self.core = ti_core.get_type_factory_instance()

    def custom_int(self, bits, signed=True):
        return self.core.get_custom_int_type(bits, signed)

    def custom_float(self,
                     significand_type,
                     exponent_type=None,
                     compute_type=None,
                     scale=1.0):
        import taichi as ti
        if compute_type is None:
            compute_type = ti.get_runtime().default_fp.get_ptr()
        return self.core.get_custom_float_type(significand_type,
                                               exponent_type,
                                               compute_type,
                                               scale=scale)
