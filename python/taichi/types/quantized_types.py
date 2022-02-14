from taichi._lib.utils import ti_core as _ti_core
from taichi.lang import impl
from taichi.types.primitive_types import i32


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


class Quant:
    """Generator of quantized types.

    For more details, read https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf.
    """
    @staticmethod
    def int(bits, signed=False, compute=None):
        """Generates a quantized type for integers.

        Args:
            bits (int): Number of bits.
            signed (bool): Signed or unsigned.
            compute (DataType): Type for computation.

        Returns:
            DataType: The specified type.
        """
        if compute is None:
            compute = impl.get_runtime().default_ip
        return type_factory.custom_int(bits, signed, compute)

    @staticmethod
    def fixed(frac, signed=True, num_range=1.0, compute=None):
        """Generates a quantized type for fixed-point real numbers.

        Args:
            frac (int): Number of bits.
            signed (bool): Signed or unsigned.
            num_range (float): Range of the number.
            compute (DataType): Type for computation.

        Returns:
            DataType: The specified type.
        """
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=i32)
        if signed:
            scale = num_range / 2**(frac - 1)
        else:
            scale = num_range / 2**frac
        if compute is None:
            compute = impl.get_runtime().default_fp
        return type_factory.custom_float(frac_type, None, compute, scale)

    @staticmethod
    def float(exp, frac, signed=True, compute=None):
        """Generates a quantized type for floating-point real numbers.

        Args:
            exp (int): Number of exponent bits.
            frac (int): Number of fraction bits.
            signed (bool): Signed or unsigned.
            compute (DataType): Type for computation.

        Returns:
            DataType: The specified type.
        """
        # Exponent is always unsigned
        exp_type = Quant.int(bits=exp, signed=False, compute=i32)
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=i32)
        if compute is None:
            compute = impl.get_runtime().default_fp
        return type_factory.custom_float(significand_type=frac_type,
                                         exponent_type=exp_type,
                                         compute_type=compute)


# Unstable API
quant = Quant

__all__ = []
