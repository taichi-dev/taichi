from taichi._lib import core as _ti_core
from taichi.lang import impl


class TypeFactory:
    """A Python-side TypeFactory wrapper."""
    def __init__(self):
        self.core = _ti_core.get_type_factory_instance()

    def custom_int(self, bits, signed=True, compute_type=None):
        """Generates a custom int type.

        Args:
            bits (int): Number of bits.
            signed (bool): Signed or unsigned.
            compute_type (DataType): Type for computation.

        Returns:
            DataType: The specified type.
        """
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
        """Generates a custom float type.

        Args:
            significand_type (DataType): Type of significand.
            exponent_type (DataType): Type of exponent.
            compute_type (DataType): Type for computation.
            scale (float): Scaling factor.

        Returns:
            DataType: The specified type.
        """
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
