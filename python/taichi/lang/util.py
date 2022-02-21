import functools
import os
import traceback

import numpy as np
from colorama import Fore, Style
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.types.primitive_types import (f16, f32, f64, i8, i16, i32, i64, u8,
                                          u16, u32, u64)

_has_pytorch = False

_env_torch = os.environ.get('TI_ENABLE_TORCH', '1')
if not _env_torch or int(_env_torch):
    try:
        import torch
        _has_pytorch = True
    except:
        pass


def has_pytorch():
    """Whether has pytorch in the current Python environment.

    Returns:
        bool: True if has pytorch else False.

    """
    return _has_pytorch


from distutils.spawn import find_executable

# Taichi itself uses llvm-10.0.0 to compile.
# There will be some issues compiling CUDA with other clang++ version.
_clangpp_candidates = ['clang++-10']
_clangpp_presence = None
for c in _clangpp_candidates:
    if find_executable(c) is not None:
        _clangpp_presence = find_executable(c)


def has_clangpp():
    return _clangpp_presence is not None


def get_clangpp():
    return _clangpp_presence


def is_taichi_class(rhs):
    taichi_class = False
    try:
        if rhs._is_taichi_class:
            taichi_class = True
    except:
        pass
    return taichi_class


def to_numpy_type(dt):
    """Convert taichi data type to its counterpart in numpy.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in numpy.

    """
    if dt == f32:
        return np.float32
    if dt == f64:
        return np.float64
    if dt == i32:
        return np.int32
    if dt == i64:
        return np.int64
    if dt == i8:
        return np.int8
    if dt == i16:
        return np.int16
    if dt == u8:
        return np.uint8
    if dt == u16:
        return np.uint16
    if dt == u32:
        return np.uint32
    if dt == u64:
        return np.uint64
    if dt == f16:
        return np.half
    assert False


def to_pytorch_type(dt):
    """Convert taichi data type to its counterpart in torch.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in torch.

    """
    # pylint: disable=E1101
    if dt == f32:
        return torch.float32
    if dt == f64:
        return torch.float64
    if dt == i32:
        return torch.int32
    if dt == i64:
        return torch.int64
    if dt == i8:
        return torch.int8
    if dt == i16:
        return torch.int16
    if dt == u8:
        return torch.uint8
    if dt == f16:
        return torch.float16
    if dt in (u16, u32, u64):
        raise RuntimeError(
            f'PyTorch doesn\'t support {dt.to_string()} data type.')
    assert False


def to_taichi_type(dt):
    """Convert numpy or torch data type to its counterpart in taichi.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in taichi.

    """
    if type(dt) == _ti_core.DataType:
        return dt

    if dt == np.float32:
        return f32
    if dt == np.float64:
        return f64
    if dt == np.int32:
        return i32
    if dt == np.int64:
        return i64
    if dt == np.int8:
        return i8
    if dt == np.int16:
        return i16
    if dt == np.uint8:
        return u8
    if dt == np.uint16:
        return u16
    if dt == np.uint32:
        return u32
    if dt == np.uint64:
        return u64
    if dt == np.half:
        return f16

    if has_pytorch():
        # pylint: disable=E1101
        if dt == torch.float32:
            return f32
        if dt == torch.float64:
            return f64
        if dt == torch.int32:
            return i32
        if dt == torch.int64:
            return i64
        if dt == torch.int8:
            return i8
        if dt == torch.int16:
            return i16
        if dt == torch.uint8:
            return u8
        if dt == torch.float16:
            return f16
        if dt in (u16, u32, u64):
            raise RuntimeError(
                f'PyTorch doesn\'t support {dt.to_string()} data type.')

    raise AssertionError(f"Unknown type {dt}")


def cook_dtype(dtype):
    if isinstance(dtype, _ti_core.DataType):
        return dtype
    if isinstance(dtype, _ti_core.Type):
        return _ti_core.DataType(dtype)
    if dtype is float:
        return impl.get_runtime().default_fp
    if dtype is int:
        return impl.get_runtime().default_ip
    raise ValueError(f'Invalid data type {dtype}')


def in_taichi_scope():
    return impl.inside_kernel()


def in_python_scope():
    return not in_taichi_scope()


def taichi_scope(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        assert in_taichi_scope(), \
                f'{func.__name__} cannot be called in Python-scope'
        return func(*args, **kwargs)

    return wrapped


def python_scope(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        assert in_python_scope(), \
                f'{func.__name__} cannot be called in Taichi-scope'
        return func(*args, **kwargs)

    return wrapped


def warning(msg, warning_type=UserWarning, stacklevel=1, print_stack=True):
    """Print a warning message. Note that the builtin `warnings` module is
    unreliable since it may be suppressed by other packages such as IPython.

    Args:
        msg (str): message to print.
        warning_type (Warning): type of warning.
        stacklevel (int): warning stack level from the caller.
        print_stack (bool): whether to print the stack
    """
    msg = f'{warning_type.__name__}: {msg}'
    if print_stack:
        msg += f'\n{get_traceback(stacklevel)}'
    print(Fore.YELLOW + Style.BRIGHT + msg + Style.RESET_ALL)


def get_traceback(stacklevel=1):
    s = traceback.extract_stack()[:-1 - stacklevel]
    return ''.join(traceback.format_list(s))


__all__ = []
