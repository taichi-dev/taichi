"""
This module defines generators of quantized types.
For more details, read https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf.
"""
from taichi._lib.utils import ti_python_core as _ti_python_core
from taichi.lang import impl
from taichi.types.primitive_types import i32

_type_factory = _ti_python_core.get_type_factory_instance()


def int(bits, signed=True, compute=None):  # pylint: disable=W0622
    """Generates a quantized type for integers.

    Args:
        bits (int): Number of bits.
        signed (bool): Signed or unsigned.
        compute (DataType): Type for computation.

    Returns:
        DataType: The specified type.
    """
    if compute is None:
        compute = impl.get_runtime().default_ip if signed else impl.get_runtime().default_up
    if isinstance(compute, _ti_python_core.DataType):
        compute = compute.get_ptr()
    return _type_factory.get_quant_int_type(bits, signed, compute)


def fixed(bits, signed=True, max_value=1.0, compute=None, scale=None):
    """Generates a quantized type for fixed-point real numbers.

    Args:
        bits (int): Number of bits.
        signed (bool): Signed or unsigned.
        max_value (float): Maximum value of the number.
        compute (DataType): Type for computation.
        scale (float): Scaling factor. The argument is prioritized over range.

    Returns:
        DataType: The specified type.
    """
    if compute is None:
        compute = impl.get_runtime().default_fp
    if isinstance(compute, _ti_python_core.DataType):
        compute = compute.get_ptr()
    # TODO: handle cases with bits > 32
    underlying_type = int(bits=bits, signed=signed, compute=i32)
    if scale is None:
        if signed:
            scale = max_value / 2 ** (bits - 1)
        else:
            scale = max_value / 2**bits
    return _type_factory.get_quant_fixed_type(underlying_type, compute, scale)


def float(exp, frac, signed=True, compute=None):  # pylint: disable=W0622
    """Generates a quantized type for floating-point real numbers.

    Args:
        exp (int): Number of exponent bits.
        frac (int): Number of fraction bits.
        signed (bool): Signed or unsigned.
        compute (DataType): Type for computation.

    Returns:
        DataType: The specified type.
    """
    if compute is None:
        compute = impl.get_runtime().default_fp
    if isinstance(compute, _ti_python_core.DataType):
        compute = compute.get_ptr()
    # Exponent is always unsigned
    exp_type = int(bits=exp, signed=False, compute=i32)
    # TODO: handle cases with frac > 32
    frac_type = int(bits=frac, signed=signed, compute=i32)
    return _type_factory.get_quant_float_type(frac_type, exp_type, compute)


__all__ = ["int", "fixed", "float"]
