from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl


class TypeFactory:
    def __init__(self):
        self.core = _ti_core.get_type_factory_instance()

    def custom_int(self, bits, signed=True, compute_type=None):
        if compute_type is None:
            compute_type = impl.get_runtime().default_ip
        if isinstance(compute_type, _ti_core.DataType):
            compute_type = compute_type.get_ptr()
        return self.core.get_custom_int_type(bits, signed, compute_type)

    def custom_float(self,
                     significand_type,
                     exponent_type=None,
                     compute_type=None,
                     scale=1.0):
        if compute_type is None:
            compute_type = impl.get_runtime().default_fp
        if isinstance(compute_type, _ti_core.DataType):
            compute_type = compute_type.get_ptr()
        return self.core.get_custom_float_type(significand_type,
                                               exponent_type,
                                               compute_type,
                                               scale=scale)


# Unstable API
type_factory = TypeFactory()
